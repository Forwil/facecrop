import web
import cv2
import face

urls = ('/upload', 'Upload')
render = web.template.render('templates/',)

filedir = './static' 

class Upload:

    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return render.upload('')

    def POST(self):
        x = web.input(myfile={})
        if 'myfile' in x: # to check if the file-object is created
            filepath = x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
            filename = filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
            fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
            fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.
            outfile = face.getMark(filename)

        return render.upload(outfile)

if __name__ == "__main__":
   app = web.application(urls, globals()) 
   app.run()
