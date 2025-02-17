import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

#from config.config import frame_capture_settings
# =========================
# Helper functions for display
# =========================

def get_yolo_boxes_from_annotation_file(annotation_file):
    boxes = []
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return boxes  # return empty list if no file
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # Each line: "class_id x_center y_center width height"
        box = line.strip().split(' ')
        if len(box) != 5:
            continue  # skip invalid lines
        box = [float(x) for x in box]
        boxes.append(box)
    return boxes

# =========================
# The interactive BoxEditor class
# Used to add and remove bounding box annotations for each frame interactively. All bounding boxes added will be associated with the target class, defined in the .env file
# =========================

class BoxEditor:
    def __init__(self, ax, fig, boxes, annotation_filename, img_width, img_height, results):
        """
        Parameters:
          ax: matplotlib axes where the image is shown.
          fig: the current figure.
          boxes: a list of boxes in YOLO format [class_id, x_center_norm, y_center_norm, width_norm, height_norm].
          annotation_path: path to the annotation file (to be updated on save).
          img_width, img_height: image dimensions in pixels.
        """
        self.ax = ax
        self.fig = fig
        self.boxes = boxes  # list of boxes in YOLO format
        self.annotation_filename = annotation_filename
        self.img_width = img_width
        self.img_height = img_height
        self.results = results
        
        # For adding new boxes (left click)
        self.temp_point = None  # stores the first corner (x, y) when left clicking
        self.temp_marker = None  # a temporary marker for the first click

        # A list to hold the drawn rectangle patches for each box
        self.patches = []
        for box in self.boxes:
            patch = self.draw_box(box)
            self.patches.append(patch)
        for result in self.results:
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    # In COCO, the 'person' class typically has an id of 0.
                    box = [cls, x1, y1, x2, y2]
                    patch = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor=(0,0,0,0), lw=1)
                    self.ax.add_patch(patch)    

        self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """Callback for mouse button press events."""
        if event.inaxes != self.ax:
            return  # ignore clicks outside the axes

      
    def draw_box(self, yolo_box, edgecolor='green'):
        """Draw a rectangle for a box in YOLO format and return the patch."""
        class_id = yolo_box[0]
        xc = yolo_box[1] * self.img_width
        yc = yolo_box[2] * self.img_height
        w = yolo_box[3] * self.img_width
        h = yolo_box[4] * self.img_height
        x0 = xc - w/2
        y0 = yc - h/2
        patch = plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=1)
        self.ax.add_patch(patch)
        self.fig.canvas.draw_idle()
        return patch
        
    def disconnect(self):
        """Disconnect the mouse click event."""
        self.fig.canvas.mpl_disconnect(self.cid)

# =========================
# Key press callback for the editor
# =========================

def on_key_editor(event, editor):
    """
    When 'n' or 'enter' is pressed, save the annotation file and close the figure.
    When 'q' is pressed, save and quit the program.
    """
    if event.key in ['n', 'enter', 'return']:
        editor.disconnect()
        plt.close()
    elif event.key == 'q':
        editor.disconnect()
        plt.close('all')
        sys.exit(0)
    else:
        # Ignore any other keys
        pass

# =========================
# Main display function (modified to include editing)
# =========================

def display_images(folder, model):
    """
    Loop through annotation .txt files in the default_label_dir folder. For each file, 
    display the corresponding image and the existing bounding boxes. 
    Then allow the user to add boxes with left-clicks and remove boxes with right-clicks.
    Press 'n' or 'enter' to save changes and move to the next image,
    or press 'q' to quit the program.
    This will move the annotation files to the edited_label_dir folder and out of the default_label_dir folder.

    """
    for annotation_filename in sorted(os.listdir(folder)):
        if not annotation_filename.lower().endswith('.txt'):
            continue
        annotation_path = os.path.join(folder, annotation_filename)
        # Load image (convert from BGR to RGB)
        img_filename = annotation_filename.replace('.txt', '.jpg')
        img_folder = str(folder).replace('labels', 'images')
        img_path = os.path.join(img_folder, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip if image cannot be loaded
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)

        # Construct the annotation file path.
        print("Loading annotations from:", annotation_path)
        boxes = get_yolo_boxes_from_annotation_file(annotation_path)

        # Create figure and axis; display the image.
        fig, ax = plt.subplots()
        ax.imshow(img)
        height, width, _ = img.shape

        # (Optional) Draw the original boxes with your existing function:
        # show_yolo_boxes(boxes, ax, height, width)

        # Create a BoxEditor to enable interactive editing.
        editor = BoxEditor(ax, fig, boxes, annotation_filename, width, height, results)

        # Connect the key press event; we pass the editor to the callback via a lambda.
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key_editor(event, editor))
        print(f"Displaying {img_path}.")
        print("  Left-click twice to add a box; right-click inside a box to remove it.")
        print("  Press 'n' or 'enter' to save and move to the next image, or 'q' to quit.")


        mng = plt.get_current_fig_manager()
        try:
            # First try with state('zoomed')
            mng.window.state('zoomed')
        except Exception:
            try:
                # If that fails, try with attributes
                mng.window.attributes('-zoomed', True)
            except Exception as e:
                print("Could not maximize the window:", e)

        # # Maximize window if possible and show the figure (this call blocks until the figure is closed)
        # mng = plt.get_current_fig_manager()
        # try:
        #     mng.window.showMaximized()
        # except AttributeError:
        #     print(mng.__dict__)
        #     pass  # Some backends may not support this

        plt.show()

# =========================
# Run the viewer/editor
# =========================
folder_path = 'datasets/dataset_webcam/labels/train'

model_path = 'finetune/yolo11l_webcam_finetune/weights/best.pt'
model = YOLO(model_path).to('cuda')
display_images(folder_path, model)
