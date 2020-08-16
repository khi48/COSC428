### Lecture Attendance Tracker

This project takes images from a Raspberry Pi and thermal camera to automatically detect how many students are attending a lecture.

By using thermal camera rather than a regular camera, no identifying features are ever taken or stored creating a safe system. All images were taken with the students consent.

By tracking how many students are attending lectures, organisers can determine how much of the lecture theatre is being utilised, and make changes accordingly.

Tracking students may also give the option to study how lecture attendance changes over the course of the year, and with different lecturers, and potentially match them up to grades.

The final result was shown to accurately predict the lecture attendance with 96%-100% accuracy.

<p align="centre">
	<img src="https://github.com/khi48/LectureAttendanceCV/blob/master/testImages/Final%20Images/finalContours.png" alt="Image of thermal camera detecting students in lecture theatre">
</p>

## Method
The initial infrared images show multiple hot spots for each person. For example, jerseys cover up the body, making it appear cool relative to the warm faces and hands, which don't appear to be connected. 

This problem was overcome by finding all the warm areas, and filtering these areas by size. The larger areas are more likely to be people. As people at the back of the theatre are relatively smaller than people at the front, this area varied with height.

<p align="centre">
	<img src="https://github.com/khi48/LectureAttendanceCV/blob/master/testImages/LeptonImages/A2_Monday13May_fullLecutre.png" alt="Image of thermal camera reading of lecture theatre">
</p>

The final method used for the person detection algorithm is outlined in the diagram below:
<p align="centre">
	<img src="https://github.com/khi48/LectureAttendanceCV/blob/master/Documents/Lecture%20Theatre%20People%20Counting.pdf" alt="Image of thermal camera reading of lecture theatre">
</p>
 