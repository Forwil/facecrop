import web
import cv2
import face
import json

urls = ('/upload', 'Upload')
urls = ('/', 'Upload')
render = web.template.render('templates/',)

filedir = './static' 

def makeMap(outfile):
    m = {}
    if len(outfile) == 2:
        m["singleImage"] = outfile[1]
        m["printImage"] = outfile[0]
    return json.dumps(m)

class Upload:

    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return render.upload('')

    def POST(self):
        x = web.input(myfile={})
        if 'myfile' in x: # to check if the file-object is created
            filepath = x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
            filename = filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
            try:
                fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
            except:
                return render.upload("")
            fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.
            height = 413
            width = 295
            color = 2
            rotate = 0
            if 'height' in x:
                height = int(x.height)
            if 'width' in x:
                width = int(x.width)
            if 'color' in x:
                color = int(x.color)
            if 'rotate' in x:
                rotate = int(x.rotate)
            outfile = face.getMark(filename,height,width,color,rotate)
#        return render.upload(outfile)
        return makeMap(outfile) 

if __name__ == "__main__":
   app = web.application(urls, globals()) 
   app.run()
