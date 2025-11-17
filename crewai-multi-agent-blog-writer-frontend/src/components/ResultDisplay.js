import { FaDownload } from "react-icons/fa";
import styled from "styled-components";

const ResultContainer = styled.div`
  margin-top: 20px;
  text-align: center;
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

const ResultText = styled.pre`
  background-color: #f8f8f8;
  padding: 15px;
  border-radius: 8px;
  white-space: pre-wrap;
  text-align: left;
  max-height: 400px;
  overflow-y: auto;
  margin-bottom: 20px;
`;

const DownloadButton = styled.button`
  padding: 12px 24px;
  background-color: #28a745;
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s ease;
  &:hover {
    background-color: #218838;
  }
`;

const ResultDisplay = ({ result, handleDownload }) => {
  if (!result) return null;

  // result.raw가 있으면 사용, 없으면 result 자체가 문자열일 수 있음
  const displayText =
    result.raw ||
    (typeof result === "string" ? result : JSON.stringify(result, null, 2));

  return (
    <ResultContainer>
      <h2>작성한 콘텐츠를 다운로드 해보세요!</h2>
      <ResultText>{displayText}</ResultText>
      <DownloadButton onClick={handleDownload}>
        <FaDownload style={{ marginRight: "8px" }} />
        다운로드
      </DownloadButton>
    </ResultContainer>
  );
};

export default ResultDisplay;
