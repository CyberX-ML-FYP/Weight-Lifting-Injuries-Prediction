import { useCallback, useRef, useState } from "react";
import { FaCloudUploadAlt } from "react-icons/fa";
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_BYTES } from "../../utils/validation";
import { formatBytes } from "../../utils/validation";

interface DropZoneProps {
  onFileSelected: (file: File) => void;
}

export default function DropZone({ onFileSelected }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      const file = fileList?.[0];
      if (!file) return;
      onFileSelected(file);
    },
    [onFileSelected],
  );

  return (
    <div
      className={`dropzone ${isDragging ? "dropzone-active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="dropzone-icon">
        <FaCloudUploadAlt />
      </div>
      <p className="dropzone-title">Drag &amp; drop your lift video here</p>
      <p className="dropzone-subtitle">or click to browse files</p>
      <div className="dropzone-meta">
        <span>Supported: MP4, AVI, MOV</span>
        <span>Max size: {formatBytes(MAX_UPLOAD_BYTES)}</span>
      </div>
    </div>
  );
}
