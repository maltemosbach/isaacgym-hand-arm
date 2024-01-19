from lang_sam import LangSAM
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import TextBox, Button
from matplotlib.text import Text
from PIL import Image
import numpy as np
import torch


class LangSamInterface:
    def __init__(
        self, 
        sam_type: str = "vit_b", 
        checkpoint_path: str = "/home/user/mosbach/tools/sam_tracking/sam_tracking/ckpt/sam_vit_b_01ec64.pth"
    ) -> None:
        
        self.lang_sam = LangSAM(sam_type, checkpoint_path)

        self.point_promts = []
        self.point_labels = []
        self.text_prompt = ""

    def build_sam_interface_figure(self, color_image_numpy, title=str) -> None:
        def show_mask(mask: np.array):
            color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            image_axis.imshow(mask_image)

        def on_point_prompt(event):
            if event.inaxes == image_axis:    
                self.point_promts.append([event.xdata, event.ydata])
                self.point_labels.append(int(event.button == MouseButton.LEFT))
                self.lang_sam.sam.set_image(color_image_numpy)
                masks, scores, logits = self.lang_sam.sam.predict(point_coords=np.array(self.point_promts).astype(np.int), point_labels=np.array(self.point_labels).astype(np.int), multimask_output=False)
                self.mask[:] = masks[0]
                image_axis.imshow(color_image_numpy)
                show_mask(self.mask)

                cols = ['green' if label == 1 else 'red' for label in self.point_labels]
                xs = [point[0] for point in self.point_promts]
                ys = [point[1] for point in self.point_promts]

                self.point_scatters.append(image_axis.scatter(xs, ys, color=cols, edgecolors='white', s=50))
                    
                sam_interface_figure.canvas.draw()

        def on_text_prompt(text):
            if text not in ['', 'Left-click to add positive marker, right-click to add negative marker.']:
                if len(self.point_promts) > 0:
                    print("Cannot use text prompt when point prompts have already been added.")
                else:
                    masks, boxes, phrases, logits = self.lang_sam.predict(Image.fromarray(color_image_numpy), text)
                    self.mask[:] = torch.any(masks, dim=0).cpu().numpy()
                    image_axis.imshow(color_image_numpy)
                    show_mask(self.mask)
                    image_axis.axis('off')
                    sam_interface_figure.canvas.draw()

        def on_hover(event):
            if event.inaxes == image_axis and event.xdata is not None and event.ydata is not None:
                if text_box.text == "":
                    text_box.set_val('Left-click to add positive marker, right-click to add negative marker.')
            else:
                if text_box.text == 'Left-click to add positive marker, right-click to add negative marker.':
                    text_box.set_val('')
            sam_interface_figure.canvas.draw()

        def on_resize(event):
            bbox = image_axis.get_position()
            ax_width = bbox.x1 - bbox.x0
            ax_x = bbox.x0

            text_prompt_axis.set_position([ax_x, 0.2, ax_width, 0.05])
            clear_axis.set_position([ax_x, 0.1, ax_width / 2, 0.075])
            submit_axis.set_position([ax_x + ax_width / 2, 0.1, ax_width / 2, 0.075])

        def on_clear_button_clicked(event):
            self.point_promts.clear()
            self.point_labels.clear()
            self.mask[:] = 0

            for s in self.point_scatters:
                s.remove()
            self.point_scatters.clear()

            image_axis.imshow(color_image_numpy)
            sam_interface_figure.canvas.draw()
            print("Clear button clicked")

        def on_submit_button_clicked(event):
            self.sam_selection_submitted = True
            print("Submit button clicked")

        self.mask = np.zeros_like(color_image_numpy[..., 0])
        sam_interface_figure = plt.figure(num=title)

        image_axis = sam_interface_figure.add_subplot(111)
        image_axis.axis('off')

        sam_interface_figure.subplots_adjust(bottom=0.2)

        text_prompt_axis = sam_interface_figure.add_axes([0.1, 0.2, 0.8, 0.05])
        text_box = TextBox(text_prompt_axis, "", initial="")
        text_box.on_submit(on_text_prompt)

        clear_axis = sam_interface_figure.add_axes([0.1, 0.1, 0.4, 0.075])
        self.clear_button = Button(clear_axis, 'Clear')
        self.clear_button.on_clicked(on_clear_button_clicked)

        submit_axis = sam_interface_figure.add_axes([0.5, 0.1, 0.4, 0.075])
        self.submit_button = Button(submit_axis, 'Submit', color='darkorange', hovercolor='orange')
        self.submit_button.label.set_color('white')
        self.submit_button.label.set_fontweight('bold')
        self.submit_button.label.set_fontsize(18)
        self.submit_button.on_clicked(on_submit_button_clicked)

        image_axis.imshow(color_image_numpy)
        image_axis.set_position([0.1, 0.3, 0.8, 0.6])

        sam_interface_figure.canvas.mpl_connect('button_press_event', on_point_prompt)
        sam_interface_figure.canvas.mpl_connect('motion_notify_event', on_hover)
        sam_interface_figure.canvas.mpl_connect('resize_event', on_resize)

        self.point_scatters = []

        plt.show(block=False)

        return sam_interface_figure

    def predict(self, color_image: np.array, title: str = "select segementation ...") -> np.array:
        self.sam_selection_submitted = False
        sam_interface_figure = self.build_sam_interface_figure(color_image, title)

        while not self.sam_selection_submitted:
            plt.pause(0.01)

        plt.close(sam_interface_figure)
        
        return self.mask
