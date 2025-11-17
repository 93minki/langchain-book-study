import styled from "styled-components";

const InputGroupWrapper = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
  flex-direction: column;
  align-items: center;
`;

const Input = styled.input`
  width: 80%;
  max-width: 500px;
  padding: 12px;
  border: 1px solid #99ccff;
  border-radius: 8px;
  margin-bottom: 10px;
  font-size: 16px;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
  &:focus {
    border-color: #3399ff;
    outline: none;
    box-shadow: 0 0 5px rgba(51, 153, 255, 0.5);
  }
`;

const Button = styled.button`
  padding: 12px 24px;
  background-color: #3399ff;
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.3s ease;
  &:hover {
    background-color: #267dcc
    transform: translateY(-2px);
  }
  &:disabled {
    background-color: #b3d1ff;
    cursor: not-allowed;
  }
`;

const InputGroup = ({ topic, handleInputChange, fetchData, loading }) => {
  return (
    <InputGroupWrapper>
      <Input
        type="text"
        placeholder="블로그 작성 주제를 입력하세요"
        value={topic}
        onChange={handleInputChange}
      />
      <Button onClick={fetchData} disabled={loading || !topic}>
        {loading ? "처리 중..." : "콘텐츠 생성"}
      </Button>
    </InputGroupWrapper>
  );
};

export default InputGroup;
