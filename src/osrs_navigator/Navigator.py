import time
import numpy
from PIL import Image
from PIL import ImageGrab
import cv2
import pyautogui
from matplotlib import pyplot as plt
from .helpers import is_inside_line, line_intersection


COMPASS_POS = (632, 54)


class Navigator:

    road_to_bank = [[(462, 293), (486,261)], [(486,261),(472,119)], [(472,119),(367,113)], [(367,113),(364,137)]]

    def __init__(self):
        pass

    def reset_view(self):
        pyautogui.moveTo(COMPASS_POS, 0.2)
        pyautogui.leftClick()

    def get_minimap_lines(self,top_left, bottom_right):
        return [[(top_left[0], top_left[1]), (top_left[0], top_left[1]+93)],
                [(top_left[0], top_left[1]), (top_left[0]+108, top_left[1])],
                [(top_left[0]+ 108, top_left[1]), ((top_left[0]+108,top_left[1]+93))],
                [(top_left[0], top_left[1] + 93), ((top_left[0] + 108, top_left[1] + 93))]]


    def get_current_minimap(self, reset):
        if reset:
            self.reset_view()
        minimap = numpy.array(ImageGrab.grab(bbox=(653,63,761,156)))
        minimap = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('Navigator_images/minimap.png', minimap)
        return minimap

    def locate(self, picture):
        img = cv2.imread('./Navigator_images/map2.png', 0)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        template = picture #cv2.imread('./Navigator_images/minimap.png', 0)
        w, h = template.shape[::-1]

        # Apply template Matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        color = (0, 0, 255)
        for line in self.road_to_bank:
            cv2.line(img, line[0], line[1], color, 2)

        #plt.subplot(121), plt.imshow(res, cmap='gray')
        #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        #plt.imshow(img, cmap='gray')
        #plt.show()

        return (top_left, bottom_right)


    def navigate(self):
        minimap = self.get_current_minimap(False)
        top_left, bottom_right = self.locate(minimap)
        border_lines = self.get_minimap_lines(top_left, bottom_right)
        intercepts = []

        for line in self.road_to_bank:
            for b_line in border_lines:
                x, y =  line_intersection(line, b_line)
                if is_inside_line((x,y), line) and is_inside_line((x,y), b_line):
                    intercepts.append((x,y))
                    print(line, b_line, "intercept: ",x ,y)
        return intercepts, top_left, bottom_right


    def show_intercepts(self, intercepts, top_left, bottom_right):
        img = cv2.imread('./Navigator_images/map2.png')
        color = (0, 255, 0)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        for point in intercepts:
            cv2.line(img, (int(point[0]), int(point[1])), (int(point[0]+1), int(point[1]+1)), color, 2)
        plt.imshow(img, cmap='gray')
        plt.show()


def main():

    for i in range(2):
        print(i)
        time.sleep(1)
        nav = Navigator()
        #tunaprint(pyautogui.mouseInfo())
    while True:
        intercepts, t, b = nav.navigate()
        nav.show_intercepts(intercepts,t ,b)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



main()