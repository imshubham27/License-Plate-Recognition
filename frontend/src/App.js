import React, { useState } from 'react';
import './App.css';

import ImageContainer from './Components/ImageContainer';
import ImageForm from './Components/ImageForm';
import Header from './Components/Header';
import Navbar from './Components/Navbar';

const App = () => {
    const [newImage, setNewImage] = useState([]);

    const handleNewImage = () => {
        setNewImage([...newImage, 'New Image'])
    }

    return (
        <div className='container-fluid lpr'>
            <div className='row align-items-center'>
				<Navbar title='License Plate Recognition'/>
			</div>
            <hr />
            <div className='row align-items-center text-center'>
            <Header heading='License Plate Recognition'/>
            </div>
            <hr/>
            <div className='row align-items-center '>
            <ImageForm handleNewImage={handleNewImage} />
            </div>
            <hr/>
            <div className='row align-items-center'>
            <ImageContainer newImage={newImage} />
            </div>
        </div>
    );
}

export default App;