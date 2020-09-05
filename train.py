import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
window.title("Face_Recogniser")
dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
 
window.geometry('1560x750')
window.configure(background='black')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message = tk.Label(window, text="CCTV-Based-Attendance-System" ,bg="Green"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="white"  ,bg="black" ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=200)

txt1 = tk.Entry(window,width=20  ,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt1.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="white"  ,bg="black"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="white"  ,bg="black"  ,height=2 ,font=('times', 15, ' bold ')) 
lbl3.place(x=400, y=400)

message1 = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=42  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message1.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="White"  ,bg="black"  ,height=2 ,font=('times', 15, ' bold ')) 
lbl3.place(x=400, y=620)


message2 = tk.Label(window, text="" ,fg="black"   ,bg="white",activeforeground = "green",width=42  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=620)
 
def clear():
    txt1.delete(0, 'end')    
    res = ""
    message1.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message1.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    ID=(txt1.get())
    Name=(txt2.get())
    if(is_number(ID) and Name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret ,img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\ "+Name +"."+ID +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
            img = cv2.flip(img, 1)
            cv2.imshow('frame',img) 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + ID +" Name : "+ Name
        row = [ID , Name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text= res)
    else:
        if(is_number(ID)):
            res = "Enter Alphabetical Name"
            message1.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message1.configure(text= res)

def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,ID = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(ID))
    recognizer.save("TrainingImageLabel\Trainer.yml")
    res = "Image Trained"
    message1.configure(text= res)
    
def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainer.yml")    
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_SIMPLEX
    col_names =  ['ID','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im = cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            ID, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['ID'] == ID]['Name'].values
                tt=str(ID)+"-"+aa
                attendance.loc[len(attendance)] = [ID,aa,date,timeStamp]                
            else:
                ID='Unknown'                
                tt=str(ID)
                if(conf > 75):
                    noOfFile=len(os.listdir("ImagesUnknown"))+1
                    cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
                cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['ID'],keep='first')
        im = cv2.flip(im, 1)
        cv2.imshow('track',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)


  
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="black"  ,bg="white"  ,width=20  ,height=2 ,activebackground = "grey" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="black"  ,bg="white"  ,width=20  ,height=2, activebackground = "grey" ,font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "grey" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "grey" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "grey" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "grey" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed by Nithin and team")
copyWrite.configure(state="disabled",fg="white"  )
copyWrite.pack(side="left")
copyWrite.place(x=500, y=680)
 
window.mainloop()

                

