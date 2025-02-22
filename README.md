
# people-counter

### **Goal**: Create a repository that can serve as a backbone for finetuning object detection models on individual web cams.

The Camden Snow Bowl hosts a [web cam](https://camdensnowbowl.com/web-cam/) that displays the conditions at the base of the mountain. I have fine tuned a YOLOv11 model on a publicly available ['person detection' dataset](https://universe.roboflow.com/titulacin/person-detection-9a6mk/dataset/16), followed by a second fine tuning step on a private dataset created through assisted labeling of frames captured from the Camden Snow Bowl.


A comparison of the two models can be seen in the below gif.

![Demo Output](demo_output2.gif)


