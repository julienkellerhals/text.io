var canvas = document.getElementById("paintCanvas") 
var button = document.getElementById("clear") 

var context = canvas.getContext("2d") 

var isDrawing = false 

canvas.width = 500 
canvas.height = 300 

context.strokeStyle = "black" 
context.lineCap = "round" 
context.lineWidth = 5 

canvas.onmousedown = function(e) {
  isDrawing = true 
  context.beginPath() 
  context.moveTo(e.pageX - this.offsetLeft, 
                 e.pageY - this.offsetTop
                ) 
  convertCanvasToImage()
}

canvas.onmousemove = function(e) {
  if (isDrawing) {
    context.lineTo(e.pageX - this.offsetLeft, 
                   e.pageY - this.offsetTop
                  )
    context.stroke()
  }
  console.log("mONGO move")
}

canvas.onmouseup = function() {
  isDrawing = false 
  context.closePath()
 }

canvas.onmouseleave = function() {
	isDrawing = false 
	context.closePath() 

}

button.onclick = function() {
  context.clearRect(0, 0, canvas.width, canvas.height) 
}

async function convertCanvasToImage() {
	console.log("mongo convert")
	var img = new Image()
	//img.src = "js/mongo.jpg"
	img.src = canvas.toDataURL("image/png", 1.0)
	console.log(img)
	//document.getElementById("jj").getContext("2d").clearRect(0, 0, document.getElementById("jj").width, document.getElementById("jj").height)
	document.getElementById("jj").getContext("2d").drawImage(img, 0, 0)
}













































const zerorpc = require("zerorpc")
let client = new zerorpc.Client()

client.connect("tcp://127.0.0.1:4242")

client.invoke("echo", "server ready", (error, res) => {
  if(error || res !== 'server ready') {
    console.error(error)
  } else {
    console.log("server is ready")
  }
})


document.getElementById("battn").onclick = function() {
  /*var img = new Image()
  img.src = canvas.toDataURL("image/png", 1.0)*/

  var img = canvas.toDataURL()

  console.log(img)

  client.invoke("save_img", img, (error, res) => {
  if(error) {
      console.error(error)
    }
 })
}