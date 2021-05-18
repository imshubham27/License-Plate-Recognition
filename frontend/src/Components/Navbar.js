import React from 'react';
import '../App.css';

const NavBar = (props) => {
   
    return (
            <nav class="navbar navbar-dark bg-dark navbar-expand-lg container-fluid">
                <div class="col">
                    <h3>{props.title}</h3>
                </div>
                <div class="col-md-auto d-flex flex-row">
                    <div class="p-2">Home</div>
                    <div class="p-2">Research Paper</div>
                    <div class="p-2">About Us</div>
                </div>
                
            </nav>
    );
};

export default NavBar;