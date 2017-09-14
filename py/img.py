from PIL import Image
import io
import codecs
import re

im = Image.open("/home/vmadmin/Schreibtisch/Mongo/py/mongo.jpg")

def show():
	im.show()


def save_img(datauri):

	imgstr = re.search(r'base64,(.*)', datauri).group(1)
	#output = open('output.png', 'wb')
	#output.write(codecs.decode(b'imgstr', 'base64'))
	#output.close()
	

	temp_img = io.StringIO(codecs.decode(b'UhEUgAAAfQAAAEsCAYAAAA1u0HIAAAVbElEQ…oECBAgQKBcQKCXj0ADBAgQIECgX0Cg9xuqQIAAAQIEygX+F+MIRjzsaFqdAAAAAElFTkSuQmCC', 'base64'))
	im = Image.open(temp_img)