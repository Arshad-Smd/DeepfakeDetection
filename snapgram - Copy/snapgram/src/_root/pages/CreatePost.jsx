// CreatePost.jsx
import React, { useContext, useState } from "react";
import PostForm from "@/components/forms/PostForm";
import { PostsContext } from "./content";

const CreatePost = () => {
  const { addPost } = useContext(PostsContext);
  const [formData, setFormData] = useState({
    title: "",
    description: "",
    image: null,
    imagePreview: null,
  });

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setFormData((prevFormData) => ({
      ...prevFormData,
      [name]: name === "image" ? files[0] : value,
    }));
  };

  const handleSubmit = (data) => {
    const { imagePreview, ...rest } = data;
    const formDataToSubmit = {
      ...rest,
      image: imagePreview, // Use the base64-encoded image data
    };
    addPost(formDataToSubmit);
    setFormData({
      title: "",
      description: "",
      image: null,
      imagePreview: null,
    });
  };

  return (
    <div className="flex flex-1">
      <div className="common-container">
        <div className="max-w-5xl flex-start gap-3 justify-start w-full">
          <img
            src="public\assets\icons\add-post.svg"
            alt="add"
            height={36}
            width={36}
          />
          <h2 className="h3-bold md:h2-bold text-left w-full">Create post</h2>
        </div>

        <PostForm
          formData={formData}
          handleChange={handleChange}
          onSubmit={handleSubmit}
        />
      </div>
    </div>
  );
};

export default CreatePost;