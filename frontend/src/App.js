import './App.scss';
import React, { useState , useRef} from 'react';

function App() {

  const [mask, setMask] = useState(true);
  

  const uploadRef = null;

  return (
    <div className="App">
        
        <div className="left">
          <div className="screen">
            
          {mask ? 
            <img alt="window" src="http://127.0.0.1:2000/face_feed" ></img>
            :
            <img alt="window" src="http://127.0.0.1:2000/video_feed" ></img>
          }
          </div>

          <div className="toggle">
              <button> <img src="./webcam.png" alt="" /></button>

              {/* <label htmlFor="filePicker" style={{ background:"grey", padding:"5px 10px" }}>
              Select File
              </label>
              <input id="filePicker" style={{visibility:"hidden"}} type={"file"}></input> */}
          </div>
        </div>
        <div className="right">
          <div className="distance" onClick={()=>setMask(false)}>
            Social Distancing Detector
          </div>
          <div className="mask" onClick={()=>setMask(true)}>
            Mask Detector
          </div>
        </div>

    </div>
  );
}

export default App;
