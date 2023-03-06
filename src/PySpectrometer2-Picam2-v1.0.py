#!/usr/bin/env python3

"""
PySpectrometer2 Les Wright 2022
https://www.youtube.com/leslaboratory
https://github.com/leswright1977

This project is a follow on from: https://github.com/leswright1977/PySpectrometer

This is a more advanced, but more flexible version of the original program. Tk Has been dropped as the GUI to allow
fullscreen mode on Raspberry Pi systems and the interface is designed to fit 800*480 screens, which seem to be a
common resolution for RPi LCD's, paving the way for the creation of a stand alone bench-top instrument.

What's new:
Higher resolution (800px wide graph)
3 row pixel averaging of sensor data
Fullscreen option for the Spectrometer graph
3rd order polynomial fit of calibration data for accurate measurement.
Improved graph labelling
Labelled measurement cursors
Optional waterfall display for recording spectra changes over time.
Key Bindings for all operations

All old features have been kept, including peak hold, peak detect, Savitsky Golay filter, and the ability to save
graphs as png and data as CSV.

For instructions please consult the readme!
"""

import cv2
import numpy as np
from specFunctions import wavelength_to_rgb, savitzky_golay, peakIndexes, readcal, writecal, background, \
    generateGraticule, Record
import base64
import argparse
from picamera2 import Picamera2

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--fullscreen", "-f", help="Fullscreen (Native 800*480)", action="store_true")
group.add_argument("--waterfall", "-w", help="Enable Waterfall (Windowed only)", action="store_true")
group.add_argument("--absorbance", "-a",
                   nargs=1,
                   help="Set up a recording for absorbance measurements between sets of two wavelengths",
                   type=float)

group.add_argument("--savitzky-golay", "-s",
                   nargs='+',
                   help="Set initial Savitzky-Golay filter smoothing",
                   type=int)
group.add_argument("--gain", "-g",
                   nargs=1,
                   help="Set initial camera gain",
                   type=int)
args = parser.parse_args()
display_fullscreen = False
display_waterfall = False
absorbance_wavelengths = []
absorbance_indices = []
if args.fullscreen:
    print("Fullscreen Spectrometer enabled")
    display_fullscreen = True
if args.waterfall:
    print("Waterfall display enabled")
    display_waterfall = True
if args.absorbance:
    if len(args.absorbance) % 2:
        raise (ValueError('Need an even number of wavelength values!'))
    absorbance_wavelengths = args.absorbance

frame_width = 800
frameHeight = 600
integrate_absorbtion = False

picam2 = Picamera2()
# need to spend more time at: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
# but this will do for now!
# min and max microseconds per frame gives framerate.
# 30fps (33333, 33333)
# 25fps (40000, 40000)

picamGain = 10.0

video_config = picam2.create_video_configuration(main={"format": 'RGB888', "size": (frame_width, frameHeight)},
                                                 controls={"FrameDurationLimits": (33333, 33333)})
picam2.configure(video_config)
picam2.start()

# Change analog gain
# picam2.set_controls({"AnalogueGain": 10.0}) #Default 1
# picam2.set_controls({"Brightness": 0.2}) #Default 0 range -1.0 to +1.0
# picam2.set_controls({"Contrast": 1.8}) #Default 1 range 0.0-32.0


title1 = 'PySpectrometer 2 - Spectrograph'
title2 = 'PySpectrometer 2 - Waterfall'
stackHeight = 320 + 80 + 80  # height of the displayed CV window (graph+preview+messages)

if display_waterfall:
    # waterfall first so spectrum is on top
    cv2.namedWindow(title2, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(title2, frame_width, stackHeight)
    cv2.moveWindow(title2, 200, 200)

if display_fullscreen:
    cv2.namedWindow(title1, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(title1, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.namedWindow(title1, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(title1, frame_width, stackHeight)
    cv2.moveWindow(title1, 0, 0)

# settings for peak detect
savpoly = 7  # savgol filter polynomial max val 15
mindist = 50  # minumum distance between peaks max val 100
thresh = 20  # Threshold max val 100

calibrate = False

click_array = []
cursorX = 0
cursorY = 0


def handle_mouse(event, x, y, flags, param):
    global click_array
    global cursorX
    global cursorY
    mouseYOffset = 160
    if event == cv2.EVENT_MOUSEMOVE:
        cursorX = x
        cursorY = y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y - mouseYOffset
        click_array.append([mouseX, mouseY])


# listen for click on plot window
cv2.setMouseCallback(title1, handle_mouse)

font = cv2.FONT_HERSHEY_SIMPLEX

intensity = [0] * frame_width  # array for intensity data...full of zeroes

hold_peaks = False  # are we holding peaks?
measure = False  # are we measuring?
record_pixels = False  # are we measuring pixels and recording clicks?
recording = False

# messages
msg1 = ""
saveMsg = "No data saved"

# blank image for Waterfall
waterfall = np.zeros([320, frame_width, 3], dtype=np.uint8)
waterfall.fill(0)  # fill black

# Go grab the computed calibration data
wavelength_data, calmsg1, calmsg2, calmsg3 = readcal(frame_width)

record = Record(wavelength_data)
# find the indices of the desired wavelengths that we want to monitor absorbance
if absorbance_wavelengths:
    absorbance_indices = list(np.searchsorted(wavelength_data, absorbance_wavelengths))
    # go one index before the first wavelength of the pair because searchsorted finds the first instance larger than the
    # given number
    for index in range(0, len(absorbance_indices), 2):
        absorbance_indices[index] -= 1
    record.slice_indices = absorbance_indices

# generate the graticule data
tens, fifties = generateGraticule(wavelength_data)

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()
    y = int((frameHeight / 2) - 40)  # origin of the vertical crop
    # y=200 	#origin of the vert crop
    x = 0  # origin of the horiz crop
    h = 80  # height of the crop
    w = frame_width  # width of the crop
    cropped = frame[y:y + h, x:x + w]
    bwimage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    rows, cols = bwimage.shape
    halfway = int(rows / 2)
    # show our line on the original image
    # now a 3px wide region
    cv2.line(img=cropped,
             pt1=(0, halfway - 2),
             pt2=(frame_width, halfway - 2),
             color=(255, 255, 255),
             thickness=1)
    cv2.line(img=cropped,
             pt1=(0, halfway + 2),
             pt2=(frame_width, halfway + 2),
             color=(255, 255, 255),
             thickness=1)

    # banner image
    decoded_data = base64.b64decode(background)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, 3)
    messages = img

    # blank image for Graph
    graph = np.zeros([320, frame_width, 3], dtype=np.uint8)
    graph.fill(255)  # fill white

    # Display a graticule calibrated with cal data
    text_offset = 12
    # vertial lines every whole 10nm
    for position in tens:
        cv2.line(img=graph,
                 pt1=(position, 15),
                 pt2=(position, 320),
                 color=(200, 200, 200),
                 thickness=1)

    # vertical lines every whole 50nm
    for position, label in fifties:
        cv2.line(img=graph,
                 pt1=(position, 15),
                 pt2=(position, 320),
                 color=(0, 0, 0),
                 thickness=2)
        cv2.putText(img=graph,
                    text=f'{label} nm',
                    org=(position - text_offset, 12),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # horizontal lines
    for i in range(320):
        if i >= 64:
            if i % 64 == 0:  # suppress the first line then draw the rest...
                cv2.line(img=graph,
                         pt1=(0, i),
                         pt2=(frame_width, i),
                         color=(100, 100, 100),
                         thickness=1)

    # Now process the intensity data and display it
    # intensity = []
    for i in range(cols):
        # data = bwimage[halfway,i] #pull the pixel data from the halfway mark
        # print(type(data)) #numpy.uint8
        # average the data of 3 rows of pixels:
        dataminus1 = bwimage[halfway - 1, i]
        datazero = bwimage[halfway, i]  # pull the pixel data from the halfway mark
        dataplus1 = bwimage[halfway + 1, i]
        data = (int(dataminus1) + int(datazero) + int(dataplus1)) / 3
        data = np.uint8(data)

        if hold_peaks:
            if data > intensity[i]:
                intensity[i] = data
        else:
            intensity[i] = data

    if display_waterfall:
        # waterfall....
        # data is smoothed at this point!!!!!!
        # create an empty array for the data
        wdata = np.zeros([1, frame_width, 3], dtype=np.uint8)
        index = 0
        for i in intensity:
            rgb = wavelength_to_rgb(round(wavelength_data[index]))  # derive the color from the wavelenthData array
            luminosity = intensity[index] / 255
            b = int(round(rgb[0] * luminosity))
            g = int(round(rgb[1] * luminosity))
            r = int(round(rgb[2] * luminosity))
            # print(b,g,r)
            # wdata[0,index]=(r,g,b) #fix me!!! how do we deal with this data??
            wdata[0, index] = (r, g, b)
            index += 1
        # bright and contrast of final image
        contrast = 2.5
        brightness = 10
        wdata = cv2.addWeighted(wdata, contrast, wdata, 0, brightness)
        waterfall = np.insert(waterfall, 0, wdata, axis=0)  # insert line to beginning of array
        waterfall = waterfall[:-1].copy()  # remove last element from array

        hsv = cv2.cvtColor(waterfall, cv2.COLOR_BGR2HSV)

    # Draw the intensity data :-)
    # first filter if not holding peaks!

    if not hold_peaks:
        intensity = savitzky_golay(intensity, 17, savpoly)
        intensity = np.array(intensity)
        intensity = intensity.astype(int)
        holdmsg = "Holdpeaks OFF"
    else:
        holdmsg = "Holdpeaks ON"

    # now draw the intensity data....
    # index=0
    for index, intense in enumerate(intensity):
        rgb = wavelength_to_rgb(round(wavelength_data[index]))  # derive the color from the wvalenthData array
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        # or some reason origin is top left.
        cv2.line(img=graph,
                 pt1=(index, 320),
                 pt2=(index, 320 - intense),
                 color=(b, g, r),
                 thickness=1)
        cv2.line(img=graph,
                 pt1=(index, 319 - intense),
                 pt2=(index, 320 - intense),
                 color=(0, 0, 0),
                 thickness=1,
                 lineType=cv2.LINE_AA)
    # index+=1

    # find peaks and label them
    text_offset = 12
    thresh = int(thresh)  # make sure the data is int.
    indexes = peakIndexes(intensity, thres=thresh / max(intensity), min_dist=mindist)
    # print(indexes)
    for i in indexes:
        height = intensity[i]
        height = 310 - height
        wavelength = round(wavelength_data[i], 1)
        cv2.rectangle(graph, ((i - text_offset) - 2, height), ((i - text_offset) + 60, height - 15), (0, 255, 255), -1)
        cv2.rectangle(graph, ((i - text_offset) - 2, height), ((i - text_offset) + 60, height - 15), (0, 0, 0), 1)
        cv2.putText(img=graph,
                    text=f'{wavelength} nm',
                    org=(i - text_offset, height - 3),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)
        # flagpoles
        cv2.line(img=graph,
                 pt1=(i, height),
                 pt2=(i, height + 10),
                 color=(0, 0, 0),
                 thickness=1)

    if measure:
        # show the cursor!
        cv2.line(img=graph,
                 pt1=(cursorX, cursorY - 140),
                 pt2=(cursorX, cursorY - 180),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.line(img=graph,
                 pt1=(cursorX - 20, cursorY - 160),
                 pt2=(cursorX + 20, cursorY - 160),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(img=graph,
                    text=f'{round(wavelength_data[cursorX], 2)} nm',
                    org=(cursorX + 5, cursorY - 165),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    if record_pixels:
        # display the points
        cv2.line(img=graph,
                 pt1=(cursorX, cursorY - 140),
                 pt2=(cursorX, cursorY - 180),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.line(img=graph,
                 pt1=(cursorX - 20, cursorY - 160),
                 pt2=(cursorX + 20, cursorY - 160),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(img=graph,
                    text=str(cursorX) + 'px',
                    org=(cursorX + 5, cursorY - 165),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)
    else:
        # also make sure the click array stays empty
        click_array = []

    if click_array:
        for data in click_array:
            mouseX = data[0]
            mouseY = data[1]
            cv2.circle(img=graph,
                       center=(mouseX, mouseY),
                       radius=5,
                       color=(0, 0, 0),
                       thickness=-1)
            # we can display text :-) so we can work out wavelength from x-pos and display it ultimately
            cv2.putText(img=graph,
                        text=str(mouseX),
                        org=(mouseX + 5, mouseY),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 0))

    # stack the images and display the spectrum
    spectrum_vertical = np.vstack((messages, cropped, graph))
    # dividing lines...
    cv2.line(img=spectrum_vertical,
             pt1=(0, 80),
             pt2=(frame_width, 80),
             color=(255, 255, 255),
             thickness=1)
    cv2.line(img=spectrum_vertical,
             pt1=(0, 160),
             pt2=(frame_width, 160),
             color=(255, 255, 255),
             thickness=1)
    # print the messages
    cv2.putText(img=spectrum_vertical,
                text=calmsg1,
                org=(490, 15),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    cv2.putText(img=spectrum_vertical,
                text=calmsg3,
                org=(490, 33),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    cv2.putText(img=spectrum_vertical,
                text=saveMsg,
                org=(490, 51),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    cv2.putText(img=spectrum_vertical,
                text=f'Gain: {picamGain}',
                org=(490, 69),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    # Second column
    cv2.putText(img=spectrum_vertical,
                text=holdmsg,
                org=(640, 15),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    cv2.putText(img=spectrum_vertical,
                text=f'Savgol Filter: {savpoly}',
                org=(640, 33),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    cv2.putText(img=spectrum_vertical,
                text=f'Label Peak Width: {mindist}',
                org=(640, 51),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    cv2.putText(img=spectrum_vertical,
                text=f'Label Threshold: {thresh}',
                org=(640, 69),
                fontFace=font,
                fontScale=0.4,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA)
    cv2.imshow(title1, spectrum_vertical)

    if display_waterfall:
        # stack the images and display the waterfall
        waterfall_vertical = np.vstack((messages, cropped, waterfall))
        # dividing lines...
        cv2.line(img=waterfall_vertical,
                 pt1=(0, 80),
                 pt2=(frame_width, 80),
                 color=(255, 255, 255),
                 thickness=1)
        cv2.line(img=waterfall_vertical,
                 pt1=(0, 160),
                 pt2=(frame_width, 160),
                 color=(255, 255, 255),
                 thickness=1)
        # Draw this stuff over the top of the image!
        # Display a graticule calibrated with cal data
        text_offset = 12

        # vertical lines every whole 50nm
        for position, label in fifties:
            for i in range(162, 480):
                if i % 20 == 0:
                    cv2.line(img=waterfall_vertical,
                             pt1=(position, i),
                             pt2=(position, i + 1),
                             color=(0, 0, 0),
                             thickness=2)
                    cv2.line(img=waterfall_vertical,
                             pt1=(position, i),
                             pt2=(position, i + 1),
                             color=(255, 255, 255),
                             thickness=1)
            cv2.putText(img=waterfall_vertical,
                        text=f'{label} nm',
                        org=(position - text_offset, 475),
                        fontFace=font,
                        fontScale=0.4,
                        color=(0, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.putText(img=waterfall_vertical,
                        text=f'{label} nm',
                        org=(position - text_offset, 475),
                        fontFace=font,
                        fontScale=0.4,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        cv2.putText(img=waterfall_vertical,
                    text=calmsg1,
                    org=(490, 15),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)
        cv2.putText(img=waterfall_vertical,
                    text=calmsg3,
                    org=(490, 33),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)
        cv2.putText(img=waterfall_vertical,
                    text=saveMsg,
                    org=(490, 51),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)
        cv2.putText(img=waterfall_vertical,
                    text=f'Gain: {picamGain}',
                    org=(490, 69),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        cv2.putText(img=waterfall_vertical,
                    text=holdmsg,
                    org=(640, 15),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        cv2.imshow(title2, waterfall_vertical)

    if recording:
        if click_array and not (len(click_array) % 2):
            record.spectrum(intensity, slice_indices=click_array)
        if absorbance_indices:
            record.absorbance(intensity)

    if integrate_absorbtion:
        integrate_absorbtion = record.calibrate_absorbance(intensity, concentration)


    key_press = cv2.waitKey(1)
    # key_press = cv2.pollKey()
    if key_press == ord('q'):
        break
    elif key_press == ord('h'):
        if not hold_peaks:
            hold_peaks = True
        elif hold_peaks:
            hold_peaks = False
    elif key_press == ord("s"):
        if display_waterfall:
            saveMsg = Record.snapshot(wavelength_data, intensity, spectrum_vertical, waterfall_vertical)
        else:
            saveMsg = Record.snapshot(wavelength_data, intensity, spectrum_vertical)
    elif key_press == ord("c"):
        calcomplete = writecal(click_array)
        if calcomplete:
            # overwrite wavelength data
            # Go grab the computed calibration data
            wavelength_data, calmsg1, calmsg2, calmsg3 = readcal(frame_width)
            # overwrite graticule data
            tens, fifties = generateGraticule(wavelength_data)
    elif key_press == ord("x"):
        click_array = []
    elif key_press == ord("m"):
        record_pixels = False  # turn off recpixels!
        if not measure:
            measure = True
        elif measure:
            measure = False
    elif key_press == ord("p"):
        measure = False  # turn off measure!
        if not record_pixels:
            record_pixels = True
        elif record_pixels:
            record_pixels = False
    elif key_press == ord("o"):  # sav up
        savpoly += 1
        if savpoly >= 15:
            savpoly = 15
    elif key_press == ord("l"):  # sav down
        savpoly -= 1
        if savpoly <= 0:
            savpoly = 0
    elif key_press == ord("i"):  # Peak width up
        mindist += 1
        if mindist >= 100:
            mindist = 100
    elif key_press == ord("k"):  # Peak Width down
        mindist -= 1
        if mindist <= 0:
            mindist = 0
    elif key_press == ord("u"):  # label thresh up
        thresh += 1
        if thresh >= 100:
            thresh = 100
    elif key_press == ord("j"):  # label thresh down
        thresh -= 1
        if thresh <= 0:
            thresh = 0

    elif key_press == ord("t"):  # Gain up!
        picamGain += 1
        if picamGain >= 50:
            picamGain = 50.0
        picam2.set_controls({"AnalogueGain": picamGain})
        print(f'Camera Gain: {picamGain}')
    elif key_press == ord("g"):  # Gain down
        picamGain -= 1
        if picamGain <= 0:
            picamGain = 0.0
        picam2.set_controls({"AnalogueGain": picamGain})
        print(f'Camera Gain: {picamGain}')

    elif key_press == ord("r"):  # Record between two wavelengths
        if recording:
            recording = False
        elif not recording:
            recording = True

    elif key_press == ord("z"):  # Record between two wavelengths
        cycles_completed = 0

    elif key_press == ord("a"):  # Record between two wavelengths
        if absorbance_indices:
            concentration = input("Concentration: ")
            integrate_absorbtion = True
# Everything done
cv2.destroyAllWindows()
