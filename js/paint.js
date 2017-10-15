$(document).ready(function(){
  var canvas = document.getElementById("paintCanvas")
  var button = document.getElementById("clear")

  var context = canvas.getContext("2d")

  var isDrawing = false

  canvas.height = $(".canvasContainer").innerHeight();
  canvas.width = $(".canvasContainer").innerWidth();

  context.strokeStyle = "black"
  context.lineCap = "round"
  context.lineWidth = 1 //5

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
    console.log("Mausbewegung festgestellt")
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
  }

  /*switch page*/

  var panelIndex = 0;


  $("#switchBtn").click(function() {
    if (panelIndex == 1) {
      panelIndex = 0;
      $("#switchBtn").text("history")
    } else {
      panelIndex = 1;
      $("#switchBtn").text("main")
    }
    switchPanel();
  });

  function switchPanel() {
    if (panelIndex == 0) {
      $(".mainPanel").css("display", "block");
      $(".historyPanel").css("display", "none");
    } else {
      $(".mainPanel").css("display", "none");
      $(".historyPanel").css("display", "block");
    }
  }
  

  /* ----- MAIN PAGE ----- */

  $("#done").click(function() {
    var possible = "A";
    var randomText = possible.charAt(Math.floor(Math.random() * possible.length));
    var element = $("<div></div>").text(randomText).hide();
    $(".elementList").append(element);
    element.fadeIn();
    $("#done").css("background-color", "black");

    $("#done").delay(1000).css("background-color", "lightgray");

    //$("#done").delay(10000000).css("background-color", "lightgray"); 

    //context.clearRect(0, 0, canvas.width, canvas.height);
  });

  $("#create").click(function() {
    var text = $(".elementList").children().text();
    var element = $("<div></div>").text(text).hide();
    $(".elementList").children().remove();
    $(".historyList").append(element);
    element.fadeIn();
    var text = "";

    $("#create").css("background-color", "gray");
    //$("#done").delay(10000000).css("background-color", "lightgray");
  });
});














































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

$("#done").click(function() {
  var canvas = document.getElementById("paintCanvas")
  var context = canvas.getContext('2d')
  var imgData = context.getImageData(0,0,canvas.width,canvas.height)
  var data = imgData.data
  var imgArray = []

  //TODO 
  for(var i=0; i<data.length; i+=4){
    /*console.log(data[i])
    console.log(data[i+1])
    console.log(data[i+2])
    console.log(data[1+3])*/
    var total = data[i] + data[i+1] + data[1+2] + data[i+3]
    imgArray[i] = total
  }
  
  client.invoke("init_network", (res) => {
    //console.log(res)
  })
  client.invoke("load_network", "../data/net.p", (res) => {
    console.log(res)
  })
  client.invoke("predict", imgArray, (res) => {
    console.log(res)
  })
})

/*document.getElementById("battn").onclick = function() {
  /*var img = new Image()
  img.src = canvas.toDataURL("image/png", 1.0)

  var img = canvas.toDataURL()

  console.log(img)

  client.invoke("save_img", img, (error, res) => {
  if(error) {
      console.error(error)
    }
 })
}
*/
