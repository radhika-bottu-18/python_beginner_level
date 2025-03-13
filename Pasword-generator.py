from tkinter import *
import random,string

import pyperclip

root=Tk()
root.geometry("400x400")
root.resizable(0,0)
root.title('Password Generator app')

Label(root,text='Password generator UI',font='arial 15 bold').pack(pady=10)
Label(root,text='Python',font="arial 15 bold").pack(side=BOTTOM)
pass_lable=Label(root,text='Password Length',font='arial 10 bold').pack()
pass_len=IntVar()
pass_str=StringVar()
length=Spinbox(root,from_=8,to_=32,textvariable=pass_len,width=15).pack()

def Generator():
    password = []
    if pass_len.get() >= 4 : 
        password.append(random.choice(string.ascii_uppercase))
        password.append(random.choice(string.ascii_lowercase))
        password.append(random.choice(string.digits))
        password.append(random.choice(string.punctuation))
     
        for i in range(pass_len.get()-4):
            password.append(random.choice(string.ascii_uppercase) + random.choice(string.ascii_lowercase) + random.choice(string.digits) + random.choice(string.punctuation))

        random.shuffle(password)
    else:
        for _ in range(pass_len.get()):
            password.append(random.choice(string.ascii_uppercase) + random.choice(string.ascii_lowercase) + random.choice(string.digits) + random.choice(string.punctuation))

    pass_str.set(''.join(password))

def copy_toclipborad():
    pyperclip.copy(pass_str.get())


Button(root,text='Generator password',command=Generator).pack(pady=5)
Entry(root,textvariable=pass_str).pack()
Button(root,text='Copy to clipboard',command=copy_toclipborad).pack(pady=5)
root.mainloop()
