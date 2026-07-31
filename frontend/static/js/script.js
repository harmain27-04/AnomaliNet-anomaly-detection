const input = document.getElementById("videoInput");

const preview = document.getElementById("preview");

if(input){

input.onchange=function(e){

const file=e.target.files[0];

if(file){

preview.src=URL.createObjectURL(file);

preview.style.display="block";

}

}

}