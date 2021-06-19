import Game from './game/Game.js'
import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';



let Q = {}
const URL = "https://teachablemachine.withgoogle.com/models/cdN4_6jX6/";
const checkpointURL = URL + "model.json"; // model topology
const metadataURL = URL + "metadata.json"; // model metadata
const noize_th = 0.5; // 해당 노이즈활률 아래 일경우만 입력을 받음

function getMask(x, y, width, height) {
	let canvas = document.createElement('canvas')
	let context = canvas.getContext('2d')
	canvas.width = width
	canvas.height = height
	document.body.appendChild(canvas);
	context.drawImage(Q.spriteImage, x, y, width, height, 0, 0, width, height)
	let array = []

	for (let x = 0; x < width; x += 1) {
		array[x] = []
		for (let y = 0; y < height; y += 1) {
			let data = context.getImageData(x, y, 1, 1).data
			let value = ' '
			if (data[0] === 255) {
				value = 'x'
			}
			array[x][y] = value
		}
	}

	document.body.removeChild(canvas)
	
	return array
}

function spriteImageLoaded() {
	// Get masks
	Q.masks = []
	Q.masks[0] = getMask(464, 1878, 156, 72) // Slime monster
	Q.masks[1] = getMask(286, 1834, 168, 116) // Flying saucer
	Q.masks[2] = getMask(630, 1870, 104, 80) // Pink Snake
	Q.masks[3] = Q.masks[2] // Yellow Snake
	Q.masks[4] = getMask(88, 1758, 188, 192) // Dragon
	Q.GAME = new Game()
}

function doubleClick(event) {
	if (document.webkitIsFullScreen) {
		document.webkitExitFullscreen()
	}else {
		document.body.webkitRequestFullscreen()
	}
}

async function init() {
	Q.width = 720
	Q.height = 576
	Q.spriteImage = new Image() 
	Q.spriteImage.addEventListener('load', spriteImageLoaded);
	Q.spriteImage.src = './assets/sprite.png'
	window.addEventListener('dblclick', doubleClick);
	
	Q.recognizer = speechCommands.create(
		"BROWSER_FFT", // fourier transform type, not useful to change
		undefined, // speech commands vocabulary feature, not useful for your models
		checkpointURL,
		metadataURL);
	await Q.recognizer.ensureModelLoaded();	
	
	Q.recognizer.listen(result => {
		let scores = result.scores;
		// console.log('check in recongnizer');
		// console.log(scores);
		let prediction = 2;
		let max = 0;
		if (scores[2] < noize_th)
		{
			for(let i=0; i< scores.length -1; i++)
			{
				if (scores[i] > max)
				{
					max = scores[i];
					prediction = i;
				} 
			}
			console.log(prediction);
			if(prediction == 2)
			{
				//노이즈
				console.log('잡음')
				// Q.GAME.my_model_predictionCallback('jump');
			}
			else if(prediction == 0)
			{
				//아~~~~~  2번째 클래스 
				console.log('jump')
				Q.GAME.my_model_predictionCallback('jump');
			}
			else
			{
				//박수 첫번째 클래스
				console.log('duck')
				Q.GAME.my_model_predictionCallback('duck');
			}
		}
		
		prediction = 0;

	}, {
		includeSpectrogram: true, // in case listen should return result.spectrogram
		probabilityThreshold: 0.75,
		invokeCallbackOnNoiseAndUnknown: true,
		overlapFactor: 0.50 // probably want between 0.5 and 0.75. More info in README
	});
}

window.addEventListener('load', init)

export default Q