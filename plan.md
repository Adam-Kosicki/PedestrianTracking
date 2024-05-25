
# Plan

## Goals
Use YOLO for pedestrian clip visualization

* We want to be able to select certain times where we see a pedestrian
* Display the clip of them
* Be able to check off boxes of characteristics (race, attire, walk/run/bike)

## YOLO Explanation
* Frame by frame it checks for whatever you set it to do. Default models which are the ones we will be working with for the time being grab cars and pedestrians frame by frame
* The problem is that YOLO does not give these detections persistent and unique id’s, and that’s where bytetracker comes in.

## Tasks
1. Get YOLO V8 Working
    ```
    https://github.com/ultralytics/ultralytics
    ```

2. Get bytetracker to work with YOLO (YOLO v8 preferred)
    ```
    https://github.com/ifzhang/ByteTrack
    ```
    ```
    https://docs.ultralytics.com/reference/trackers/byte_tracker/
    ```
    ```
    https://www.youtube.com/watch?v=OS5qI9YBkfk&ab_channel=Roboflow
    ```
    ```
    https://docs.ultralytics.com/guides/object-counting/#real-world-applications
    ```
    * Use bytetracker for object-counting

3. Use FFMPEG to get clips of an object
    ```
    https://docs.google.com/document/d/1480ka6sRHQeZ5Q2ZxSJV_rJJkZuTbPi6qoVIRW56Kbg/edit?usp=sharing
    ```

4. Determine some "region of interest" in which to focus? Or maybe ask user to select region of interest? 

5. Create GUI that allows the user to get these clips
    * Give them checkboxes to fill out and save for certain events

6. Make YOLO utilize only GPU (optional)

7. Create custom model for YOLO (optional)