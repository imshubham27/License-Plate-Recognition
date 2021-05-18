import React, { useState, useEffect } from 'react';
import '../App.css';
import axios from 'axios';

const API_URL = "http://localhost:5000/";

const ImageContainer = ({ newImage }) => {
    const [images, setImages] = useState([]);
    const [fallback, setFallback] = useState('');
    const [text, setText] = useState('');

    const getImages = async () => {
        try {
            const res = await axios.get(API_URL + 'images');
            if (!res.data.files) {
                setFallback(res.data.msg);
                return;
            } else {
                setImages(res.data.files);
                setText(res.data.files);
            }
        } catch (err) {
            console.log(err.messsage);
        }
    }

    useEffect(() => {
        getImages();
    }, [newImage]);

    const configImage = image => {
        return API_URL + image;
    }


    return (
        <div>
            {images.length > 0 ?
                (
                    images.map((image, index) => (
                        <div class="card-deck cd">
                            <div class="col-md-auto" style={{ width: "18rem" }} >
                                <img class="card-img-top img" src={configImage(image)} key={image} alt={image} />
                                <div class="card-body" >
                                    <h5 class="card-title name"  >{text[index]}</h5>
                                </div>
                            </div>
                        </div>
                       
                    )
                    )
                )
                :
                <>
                    <h1>
                        {fallback}
                    </h1>
                </>
            }
        </div>
    )
}

export default ImageContainer;
