import logo from './logo.svg';
import './App.css';
import React, { useEffect, useState, Fragment } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';
import TasteImg from "./TasteBot.avif";
import Form from 'react-bootstrap/Form';
//import { Img } from 'react-image';



import Post from "./components/Post";
import Header from "./components/Header";
import LeftCard from "./components/LeftCard";
import { Container, Row, Col } from "react-bootstrap";
import Button from 'react-bootstrap/Button';

function App() {
  
  const [data, setData] = useState({});
  const [backendString, setBackendString] = useState('');


  useEffect(() => {
      fetch('/api/data')
          .then(response => response.json())
          .then(data => setData(data))
          .catch(error => console.error('Error:', error));
  }, []);

  const [tfidfResults, setTfidfResults] = useState([]);
  const [query, setQuery] = useState('');

  const handleButtonClick = async () => {
    console.log('Button clicked');
  
    try {
      const response = await fetch(`http://127.0.0.1:5000/TFIDF?query=${query}`, {
        method: 'GET', // Specify the request method
      });
      console.log(response)
      const data = await response.json();
      console.log("hi")

      setTfidfResults(data.results);
    } catch (error) {
      console.error('Error fetching TFIDF results:', error);
    }
  };

  const [Word2VecResults, setWord2VecResults] = useState([]);
  const [query1, setQuery1] = useState('');

  const handleButtonClick1 = async () => {
    console.log('Button clicked');
  
    try {
      const response = await fetch(`http://127.0.0.1:5000/word2vec?query=${query1}`, {
        method: 'GET', // Specify the request method
      });
      console.log(response)
      const data = await response.json();

      setWord2VecResults(data.results);
    } catch (error) {
      console.error('Error fetching TFIDF results:', error);
    }
  };
  

  const [rankingResults, setRankingResults] = useState([]);

  const handleRankingButtonClick2 = async () => {
    try {
      const response = await fetch(`http://localhost:5000/ranking`);
      const data = await response.json();
      setRankingResults(data.results);
    } catch (error) {
      console.error('Error fetching ranking results:', error);
    }
  };



  return (
  
  <div style={{backgroundColor: 'grey', position:"absolute", left:"0",right:"0", height:"100%"}}>
    <Navbar bg="dark" variant="dark" expand="lg">
    <img src={TasteImg} height = "30sv" className="square bg-primary rounded-circle mx-2"  alt="Description" />
    <Navbar.Brand href="#home">The Taste Bot</Navbar.Brand>
    <Navbar.Toggle aria-controls="basic-navbar-nav" />
    <Navbar.Collapse id="basic-navbar-nav">
      <Nav className="ml-auto">
        <Nav.Link href="#home">Home</Nav.Link>
        <Nav.Link href="#features">Features</Nav.Link>
        <Nav.Link href="#contact">Contact</Nav.Link>
      </Nav>
    </Navbar.Collapse>
    </Navbar>
    <div class="text-center">
      <img src={TasteImg} height = "150sv" className="square bg-primary rounded-circle mx-2 mt-3"  alt="Description" />
      <h1 className="mt-2" style={{"pointer-events":"none"}}>Welcome To The TasteBot!</h1>
      <Form>
        <Form.Group className="mb-3 mx-5" bg="dark" controlId="exampleForm.ControlTextarea1">
          <Form.Control as="textarea" selectionColor={'green'} small rows={5} style={{background:"silver", cursor: "auto", "user-select":"text" }}/>
        </Form.Group>

      </Form>
    </div>
    <div>
    <label>
        Enter your query:
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>

      <button onClick={handleButtonClick}>
        Fetch TFIDF Results
      </button>

      <ul>
        {tfidfResults.map((result, index) => (
          <li key={index}>{result}</li>
        ))}
      </ul>
    </div>

    <div>
    <label>
        Enter your query:
        <input
          type="text"
          value={query1}
          onChange={(e) => setQuery1(e.target.value)}
        />
      </label>

      <button onClick={handleButtonClick1}>
        Fetch Word2VecResults Results
      </button>

      <ul>
        {Word2VecResults.map((result, index) => (
          <li key={index}>{result}</li>
        ))}
      </ul>
    </div>

    <div>
      <button onClick={handleRankingButtonClick2}>
        Get Ranking
      </button>

      <ul>
        {rankingResults.map((result, index) => (
          <li key={index}>{result}</li>
        ))}
      </ul>
    </div>
  </div>
    // <div style={{backgroundColor: 'grey', position:"absolute", left:"0",
    // right:"0", height:"100%"}}>
    //   <Navbar expand="lg" className="bg-body-tertiary">
    //   <Container>
    //     <Navbar.Brand href="#home">The Taste Bot</Navbar.Brand>
    //     <Navbar.Toggle aria-controls="basic-navbar-nav" />
    //     <Navbar.Collapse id="basic-navbar-nav">ds</Navbar.Collapse>
    //     <Nav className="me-auto">
    //       <Nav.Link href="#home">Home</Nav.Link>
    //       <Nav.Link href="#link">Link</Nav.Link>
    //       <NavDropdown title="Dropdown" id="basic-nav-dropdown">
    //         <NavDropdown.Item href="#action/3.1">Action</NavDropdown.Item>
    //         <NavDropdown.Item href="#action/3.2">
    //           Another action
    //         </NavDropdown.Item>
    //         <NavDropdown.Item href="#action/3.3">Something</NavDropdown.Item>
    //         <NavDropdown.Divider />
    //         <NavDropdown.Item href="#action/3.4">
    //           Separated link
    //         </NavDropdown.Item>
    //       </NavDropdown>
    //     </Nav>
    //   </Container>
    //   </Navbar> 
    //   <div class="text-center">
    //     <p className="fs-1">hey</p>
    //   </div>
    // </div>
  );
}

export default App;
