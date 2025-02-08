import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

from config.config import frame_capture_settings
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
# =========================

class BoxEditor:
    def __init__(self, ax, fig, boxes, annotation_filename, img_width, img_height):
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

        # For adding new boxes (left click)
        self.temp_point = None  # stores the first corner (x, y) when left clicking
        self.temp_marker = None  # a temporary marker for the first click

        # A list to hold the drawn rectangle patches for each box
        self.patches = []
        for box in self.boxes:
            patch = self.draw_box(box)
            self.patches.append(patch)
        
        # Connect the mouse click event
        self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

    def draw_box(self, yolo_box):
        """Draw a rectangle for a box in YOLO format and return the patch."""
        class_id = yolo_box[0]
        xc = yolo_box[1] * self.img_width
        yc = yolo_box[2] * self.img_height
        w = yolo_box[3] * self.img_width
        h = yolo_box[4] * self.img_height
        x0 = xc - w/2
        y0 = yc - h/2
        edgecolor = 'green' if class_id == 0 else 'red'
        patch = plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2)
        self.ax.add_patch(patch)
        self.fig.canvas.draw_idle()
        return patch

    def on_click(self, event):
        """Callback for mouse button press events."""
        if event.inaxes != self.ax:
            return  # ignore clicks outside the axes

        if event.button == 1:  # Left click: add box
            self.handle_left_click(event)
        elif event.button == 3:  # Right click: remove box
            self.handle_right_click(event)

    def handle_left_click(self, event):
        if self.temp_point is None:
            # Save the first corner and mark it
            self.temp_point = (event.xdata, event.ydata)
            self.temp_marker = self.ax.scatter(event.xdata, event.ydata, color='yellow', marker='o')
            self.fig.canvas.draw_idle()
        else:
            # Second left click: define the opposite corner
            x1, y1 = self.temp_point
            x2, y2 = event.xdata, event.ydata
            # Determine top-left and bottom-right coordinates
            x0 = min(x1, x2)
            y0 = min(y1, y2)
            x1_ = max(x1, x2)
            y1_ = max(y1, y2)
            w = x1_ - x0
            h = y1_ - y0
            # Ignore very small boxes
            if w < 1 or h < 1:
                self.clear_temp()
                return
            # Convert the pixel coordinates into YOLO format (normalized)
            xc = x0 + w/2
            yc = y0 + h/2
            x_center_norm = xc / self.img_width
            y_center_norm = yc / self.img_height
            w_norm = w / self.img_width
            h_norm = h / self.img_height
            new_box = [0, x_center_norm, y_center_norm, w_norm, h_norm]
            # Append the new box and draw it
            self.boxes.append(new_box)
            patch = self.draw_box(new_box)
            self.patches.append(patch)
            self.clear_temp()

    def clear_temp(self):
        """Clear the temporary first click marker."""
        if self.temp_marker:
            self.temp_marker.remove()
            self.temp_marker = None
        self.temp_point = None
        self.fig.canvas.draw_idle()

    def handle_right_click(self, event):
        """Remove a box if the click is inside one."""
        click_x, click_y = event.xdata, event.ydata
        idx_to_remove = None
        # Iterate through boxes and their patches
        for idx, box in enumerate(self.boxes):
            xc = box[1] * self.img_width
            yc = box[2] * self.img_height
            w = box[3] * self.img_width
            h = box[4] * self.img_height
            x0 = xc - w/2
            y0 = yc - h/2
            if x0 <= click_x <= x0 + w and y0 <= click_y <= y0 + h:
                idx_to_remove = idx
                break
        if idx_to_remove is not None:
            # Remove the box and its patch
            self.boxes.pop(idx_to_remove)
            patch = self.patches.pop(idx_to_remove)
            patch.remove()
            self.fig.canvas.draw_idle()

    def disconnect(self):
        """Disconnect the mouse click event."""
        self.fig.canvas.mpl_disconnect(self.cid)

    def save_annotations(self):
        """Save the current list of boxes (in YOLO format) to the annotation file."""
        default_annotation_filepath = os.path.join(frame_capture_settings.default_label_dir, self.annotation_filename)
        edited_annotation_filepath = os.path.join(frame_capture_settings.edited_label_dir, self.annotation_filename)
        with open(edited_annotation_filepath, 'w') as f:
            for box in self.boxes:
                line = ' '.join(str(x) for x in box)
                f.write(line + '\n')
            print(f"Annotations saved to: {edited_annotation_filepath}")
            if os.path.exists(default_annotation_filepath):
                os.remove(default_annotation_filepath)
                print(f"{default_annotation_filepath} deleted successfully.")
            else:
                print(f"{default_annotation_filepath} not found.")

# =========================
# Key press callback for the editor
# =========================

def on_key_editor(event, editor):
    """
    When 'n' or 'enter' is pressed, save the annotation file and close the figure.
    When 'q' is pressed, save and quit the program.
    """
    if event.key in ['n', 'enter', 'return']:
        editor.save_annotations()
        editor.disconnect()
        plt.close()
    elif event.key == 'q':
        editor.save_annotations()
        editor.disconnect()
        plt.close('all')
        sys.exit(0)
    else:
        # Ignore any other keys
        pass

# =========================
# Main display function (modified to include editing)
# =========================

def display_images(folder):
    """
    For each .jpg image in the folder, display it along with its current annotations.
    Then allow the user to add boxes with left-clicks and remove boxes with right-clicks.
    Press 'n' or 'enter' to save changes and move to the next image,
    or press 'q' to quit the program.
    """
    for annotation_filename in sorted(os.listdir(folder)):
        if not annotation_filename.lower().endswith('.txt'):
            continue
        annotation_path = os.path.join(folder, annotation_filename)
        # Load image (convert from BGR to RGB)
        img_filename = annotation_filename.replace('.txt', '.jpg')
        img_path = os.path.join(frame_capture_settings.img_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip if image cannot be loaded
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        editor = BoxEditor(ax, fig, boxes, annotation_filename, width, height)

        # Connect the key press event; we pass the editor to the callback via a lambda.
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key_editor(event, editor))
        print(f"Displaying {img_path}.")
        print("  Left-click twice to add a box; right-click inside a box to remove it.")
        print("  Press 'n' or 'enter' to save and move to the next image, or 'q' to quit.")

        # Maximize window if possible and show the figure (this call blocks until the figure is closed)
        mng = plt.get_current_fig_manager()
        try:
            mng.window.showMaximized()
        except AttributeError:
            pass  # Some backends may not support this

        plt.show()

# =========================
# Run the viewer/editor
# =========================

display_images(frame_capture_settings.default_label_dir)
