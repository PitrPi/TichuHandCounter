import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.uix.label import Label


class TichuHandCounter(App):
    def build(self):
        return Label(text='Hello world')


if __name__ == '__main__':
    TichuHandCounter().run()
