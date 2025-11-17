import axios from "axios";
import { useState } from "react";
import styled from "styled-components";
import InputGroup from "./components/InputGroup";
import LoadingSpinner from "./components/LoadingSpinner";
import ResultDisplay from "./components/ResultDisplay";

const AppContainer = styled.div`
  padding: 40px 20px;
  max-width: 800px;
  margin: 0 auto;
  font-family: "Helvetica Nenu", Arial, sans-serif;
`;

const Header = styled.h1`
  text-align: center;
  color: #333;
  font-size: 2.5em;
  margin-bottom: 20px;
`;

const Description = styled.p`
  text-align: center;
  margin-bottom: 40px;
  color: #666;
  font-size: 1.2em;
`;

const Error = styled.div`
  color: red;
  text-align: center;
  margin-top: 20px;
  font-size: 1.1em;
`;

function App() {
  const [topic, setTopic] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleInputChange = (e) => {
    setTopic(e.target.value);
  };

  const fetchData = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await axios.post("http://localhost:8000/crewai", {
        topic: topic,
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response ? err.response.data.error : err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result || !result.raw) return;
    const element = document.createElement("a");
    const file = new Blob([result.raw], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = "raw_data.txt";
    document.body.appendChild(element);
    element.click();
  };

  return (
    <AppContainer>
      <Header>CrewAI 블로그 콘텐츠 생성기</Header>
      <Description>
        주제에 맞는 블로그 콘텐츠를 생성하기 위해 CrewAI의 강력한 백엔드를
        사용하세요.
      </Description>
      {loading ? (
        <LoadingSpinner />
      ) : (
        <>
          <InputGroup
            topic={topic}
            handleInputChange={handleInputChange}
            fetchData={fetchData}
            loading={loading}
          />
          {error && <Error>{error}</Error>}
          <ResultDisplay result={result} handleDownload={handleDownload} />
        </>
      )}
    </AppContainer>
  );
}

export default App;
