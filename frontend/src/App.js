import './App.scss';

function App() {
  return (
    <div className="App">
        
        <div className="left">
          <div className="screen">
            {/* <iframe src="http://127.0.0.1:2000/" frameborder="0"></iframe> */}
            <img src="http://127.0.0.1:2000/face_feed" ></img>
          </div>
        </div>
        <div className="right">
          <div className="distance">
            Social Distancing Detector
          </div>
          <div className="mask">
            Mask Detector
          </div>
        </div>

    </div>
  );
}

export default App;
