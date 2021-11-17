// const electron = require('electron');
// const app = electron.app;
// const BrowserWindow = electron.BrowserWindow;
// let mainWindow;

// app.commandLine.appendSwitch ("disable-http-cache");
// function createWindow() {
// mainWindow = new BrowserWindow({
//         width: 1024, 
//         height: 768,
//         webPreferences:{
//             nodeIntegration:true,
//             webSecurity: false
//         },
//         resizable:false
//     });
// mainWindow.loadURL('http://localhost:65000');

// mainWindow.on('closed', function () {
//         mainWindow = null
//     })
// }
// app.on('ready', createWindow);
// app.on('window-all-closed', function () {
//     if (process.platform !== 'darwin') {
//         app.quit()
//     }
// });
// app.on('activate', function () {
//     if (mainWindow === null) {
//         createWindow()
//     }
// });


const electron = require('electron');
const app = electron.app;
const BrowserWindow = electron.BrowserWindow;

const path = require('path');
const url = require('url');
const isDev = require('electron-is-dev');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1024, 
    height: 768,
    webPreferences:{
        webSecurity: false
    },
    resizable:false
});
  mainWindow.loadURL(isDev ? 'http://localhost:3000' : `file://${path.join(__dirname, '../build/index.html')}`);
  mainWindow.on('closed', () => mainWindow = null);
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});