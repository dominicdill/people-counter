
# people-counter

### **Goal**: Create a repository that can serve as a backbone for finetuning object detection models on individual web cams so that they can be used to count objects.
This repo currently only supports counting of one object, as the workflow for labelling your own dataset does not allow for multiple classes. However, if another tool is used, like CVAT, then this repository could be used for fine tuning an object detection model to count multiple objects from a web cam.

If you already have Conda installed you can build and activate a suitable environment by executing 
```bash
bash create_env.sh 
```
You will be asked if you would like to download the CUDA enabled PyTorch libraries. Respond Y if you have a GPU that supports them.

Once you have the environment activated, you can utilize this repo to finetune a model on your webcam by following these steps:

1) Update the `src/config/.env`, specifically:
    * The `webcam_url` parameter should point to your url. The url should open up a video stream that does not require any user interaction (like pressing play). 
    * The `target_label` parameter should be updated if you want to detect something other than people. Look through the classes listed in `default_classes.txt`. If one of the default labels matches what you are trying to detect then update `target_label` to that label. If nothing matches, then set this parameter to the new label of your choice.
    * The `frame_interval` parameter determines the frequency at which frames from your webcam are saved to your device.
    * The `duration_hours` parameter sets the number of hours frames are captured when executing `src/capture_finetuning_images.py`. 
    * The `model_path` parameter sets the pretrained model to be used as a starting point. I suggest using a smaller model, like YOLOv11n, if your compute is limited or you need to run inference very quickly. If accuracy is of the utmost importance and you have the compute then I suggest starting with something like YOLOv11l. Pretrained base models can be found in `src/models`. If you have already fine tuned a model and wish to continue training it then you should change your model path to point to your fine tuned model. Finetuned models will be saved with the name `best.pt` and located in a subfolder of a folder with the same title as the `project_name` parameter.  

    These settings are used to determine how you collect frames to finetune on. Ideally, you will have a dataset with at least a few hundred frames that contains a wide array of conditions (like night time / daytime photos if you want your model to work in both conditions, crowded and sparse conditions, stormy and nice weather, etc.) When collecting frames for the Camden Snowbowl webcam model, I made sure to execute `src/capture_finetuning_images.py` during the morning, afternoon, and evening, as well as during snowstorms and sunny days.
2) Execute `src/capture_finetuning_images.py` over a timespan long enough to collect a good training dataset. This will likely require you to manually execute this script multiple times, during different weather conditions and times of the day. The number of frames saved from each execution will depend on your `frame_interval` and `duration_hours` values. Remember that you must manually label each and every one of these frames, so only capture as many images as you are comfortable annotating. Frames captured by this script will be saved to `images_for_manual_labeling/images` and annotation files will be saved to `images_for_manual_labeling/labels/model_default`. Annotation files will be generated using the model set by your `model_path` parameter in the `src/config/.env` file.

3) Once frames are captured from your web cam you can begin manually labeling them. You do not need to wait until you have captured all of your frames to begin labelling them. You can perform this step in parallel with step 2. To begin, execute `src/manual_train_annotation.py`. This should display an image from the `images_for_manual_labeling` directory (specifically, images that don't have an associated annotation file in the `images_for_manual_labeling/labels/manually_labeled` folder) with bounding boxes for objects detected by the model used when you ran `src/capture_finetuning_images.py`. If the bounding box positioned poorly or it detects an object not of interest to you then you should remove it by right clicking inside of it. To create new bounding boxes, simply left click twice two define the corners of your box. Press enter when you are done labelling all of the objects of interest in your frame to saved your annotation file and move on to the next frame. Pressing enter will move the annotation file from the `images_for_manual_labeling/labels/model_defaults` to the ``images_for_manual_labeling/labels/manually_labelled` folder. If you inadvertently press enter and want to re-edit a file using this tool then you will have to move that file back into the `images_for_manual_labeling/labels/model_defaults` folder.

4) 


The Camden Snow Bowl hosts a [web cam](https://camdensnowbowl.com/web-cam/) that displays the conditions at the base of the mountain. I have fine tuned a YOLOv11 model on a publicly available ['person detection' dataset](https://universe.roboflow.com/titulacin/person-detection-9a6mk/dataset/16), followed by a second fine tuning step on a private dataset created through assisted labeling of frames captured from the Camden Snow Bowl.


A comparison of the two models can be seen in the below gif.

![Demo Output](demo/demo_output2.gif)


