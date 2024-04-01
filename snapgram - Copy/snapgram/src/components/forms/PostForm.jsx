import React, { useState } from "react";

const PostForm = ({ formData, handleChange, onSubmit }) => {
  const [mediaPreview, setMediaPreview] = useState(null);
  const [showPopup, setShowPopup] = useState(false);
  const [title, setTitle] = useState();
  const [description, setDescription] = useState();

  const handleMediaChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setMediaPreview(reader.result);
        handleChange(e);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleTitleChange = (e) => {
    setTitle(e.target.value);
  }

  const handleDescriptionChange = (e) => {
    setDescription(e.target.value);
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    const fileInput = e.target.media;
    const file = fileInput.files[0];
    console.log(file);
    const formData = new FormData();

    
    
    if (file) {
      const fileType = file.type.split("/")[0]; // Get the file type (image, audio, video)
      formData.append(fileType, file);
    }
    console.log(formData, true);
    
    let name;

    try {
      // Call the API
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      // Log the response
      const data = await response.json();
      name = data[1];
      console.log("Response from server:", data);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
    //
    if (name === "fake") {
      setShowPopup(true);
    } else {
      onSubmit({ ...formData, mediaPreview, title: title, description: description });
      setMediaPreview(null);
    }
  };

  const handlePopupCancel = () => {
    setShowPopup(false);
    setMediaPreview(null);
  };

  return (
    <>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div>
          <label htmlFor="title" className="text-white font-semibold mb-1">
            Title
          </label>
          <input
            type="text"
            name="title"
            id="title"
            value={title}
            onChange={handleTitleChange}
            placeholder="Enter post title"
            className="w-full px-4 py-2 rounded-md bg-gray-700 text-white shad-input"
            style={{ width: "100%", height: "40px" }}
          />
        </div>

        <div>
          <label
            htmlFor="description"
            className="text-white font-semibold mb-1"
          >
            Description
          </label>
          <textarea
            name="description"
            id="description"
            value={description}
            onChange={handleDescriptionChange}
            placeholder="Enter post description"
            className="w-full px-4 py-2 rounded-md bg-gray-700 text-white resize-none shad-input"
            style={{ width: "100%", height: "100px" }}
            rows="4"
          ></textarea>
        </div>

        <div>
          <label htmlFor="media" className="text-white font-semibold mb-1">
            Media (Image, Audio, or Video)
          </label>
          <input
            type="file"
            name="media"
            id="media"
            onChange={handleMediaChange}
            accept="image/, audio/, video/*"
            className="w-full px-4 py-2 rounded-md bg-gray-700 text-white shad-input"
            style={{ width: "100%", height: "40px" }}
          />
          {mediaPreview && (
            <div>
              {mediaPreview.startsWith("data:image/") && (
                <img
                  src={mediaPreview}
                  alt="Preview"
                  className="mt-2 max-w-full h-auto"
                />
              )}
              {mediaPreview.startsWith("data:audio/") && (
                <audio controls>
                  <source src={mediaPreview} type="audio/mpeg" />
                </audio>
              )}
              {mediaPreview.startsWith("data:video/") && (
                <video controls>
                  <source src={mediaPreview} type="video/mp4" />
                </video>
              )}
            </div>
          )}
        </div>

        <button
          type="submit"
          className=" text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors duration-300"
          style={{ backgroundColor: "#877EFF" }}
        >
          Create Post
        </button>
      </form>

      {showPopup && (
        <div className="fixed inset-0 flex items-center justify-center bg-gray-800 bg-opacity-50">
          <div className="rounded-lg p-8" style={{ backgroundColor: "black" }}>
            <p className="text-white-800 mb-4">
              <span style={{ color: "#877EFF", fontWeight: "bolder" }}>
                DEEPFAKE DETECTED:
              </span>
              <br /> The file you've uploaded exhibits characteristics of a
              deepfake, potentially manipulated or fabricated.
              <br /> Proceed with caution and verify its authenticity before
              dissemination or use.
            </p>
            <button
              onClick={handlePopupCancel}
              className="px-4 py-2  text-white rounded-md"
              style={{ backgroundColor: "#877EFF" }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default PostForm;
