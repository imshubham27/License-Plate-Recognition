import React from 'react';

const Header = ({heading}) => {
    return (
        <div className='col title '>
			<h1>{heading}</h1>
            <h4>Upload Image</h4>
		</div>

    );
};

export default Header;


