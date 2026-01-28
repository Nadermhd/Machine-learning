#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:19:56 2023

@author: nadermhd
"""

from tkinter import *

root = Tk()
root.title("Codemy.com - Rounded Buttons")
#root.iconbitmap('c:/gui/codemy.ico')
#root.geometry("400×400")

def thing():
    my_label.config(text="You clicked the button...")

login_btn = PhotoImage(file= "123.png")
                        
img_label = Label (image=login_btn)

my_button = Button(root, image=login_btn, command=thing, borderwidth=10)
my_button.pack(pady=20)

my_label = Label (root, text="")
my_label.pack(pady=20)

mainloop()