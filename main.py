from kivymd.app import MDApp # This import creates the screen
from kivymd.uix.label import MDLabel


class App(MDApp):

    def build(self):
        label = MDLabel(text='Hello world', halign='center')
        return label
    # MANNN I HATE HASHTAGS


App().run()
