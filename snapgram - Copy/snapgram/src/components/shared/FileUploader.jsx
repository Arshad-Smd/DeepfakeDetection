import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "../ui/button";

const FileUploader = ({fieldchange,mediaUrl}) => {

  const [file, setfile] = useState([])
  const [fileurl, setfileurl] = useState("");

  const onDrop = useCallback((acceptedFiles) => {
    setfile(acceptedFiles);
    fieldchange(acceptedFiles);
    setfileurl(URL.createObjectURL(acceptedFiles[0]))
    // Do something with the files
  }, [file]);
  const { getRootProps, getInputProps } = useDropzone({ onDrop, accept: { 'image/*':['.png','.jpeg','.jpg','.svg']} });
  return (
    <div
      {...getRootProps()}
      className="flex flex-center flex-col bg-dark-3 rounded-xl cursor-pointer"
    >
      <input {...getInputProps()} className="cursor-pointer" />
      {fileurl ? (
        <>
        <div className="flex flex-1 justify-center w-full p-5 lg:p-10">
            <img src={fileurl} alt="image" className="file_uploader-img" />
            
        </div>
        <p>Click or drag photo to replace</p>
        </>
      ) : (
        <div className="file_uploader-box">
          <img
            src="public\assets\icons\file-upload.svg"
            alt="file-upload"
            height={77}
            width={96}
          />
          <h3 className="base-medium text-light-2 mb-2 mt-6">
            Drag photo here
          </h3>
          <p className=" text-light-4 small-regular mb-6">SVG,PNG,JPEG</p>
          <Button className="shad-button_dark_4">Select from computer</Button>
        </div>
      )}
    </div>
  );
};

export default FileUploader;
