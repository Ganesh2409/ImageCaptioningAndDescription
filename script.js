document.getElementById("uploadForm").onsubmit = async function (event) {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById("file");
    formData.append("file", fileInput.files[0]);

    const response = await fetch("/upload", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("description").innerText = result.description;
    document.getElementById("preview").src = result.image_url;
};
