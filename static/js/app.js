const mic = require('mic');
var recordbutton = document.getElementbyId('recording')
recordbutton.addEventListener('click',startMicrophone)
function startMicrophone() {
    var microphone = mic({
        rate: '16000',
        channels: '1',
        debug: false,
        fileType: 'wav'
    });

    console.log(microphone.getAudioStream())

    microphone.start()

}