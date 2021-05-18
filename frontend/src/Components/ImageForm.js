import React, { useState } from 'react';
import axios from 'axios';

const ImageForm = ({ handleNewImage }) => {
    const [image, setImage] = useState('');

    const handleImageUpload = e => {
        setImage(e.target.files[0]);
    }

    const handleSubmit =async (e) => {
        e.preventDefault();
        const formData = new FormData()
        formData.append('file', image)
        axios.post("http://localhost:5000/upload", formData, {
        }).then(res => {
            console.log(res);
        })
        axios.get("http://localhost:5000/py", {
            }).then(res => {
                console.log(res);
                window.location.reload();
            })
        setImage(false);
        handleNewImage();
    }


    return (
        <div>
            <div class="row justify-content-center">
                <div class="col-auto">
                    <form onSubmit={handleSubmit}>
                        <div class="row">
                            <div class="col-auto">
                                <input type="file" class="form-control" name="file" id="formFile" onChange={handleImageUpload} />
                            </div>
                            <div class="col-auto">
                                <button className="btn btn-warning" type="submit">Upload</button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

    )

}

export default ImageForm;