import styled from "styled-components";
import React from 'react';
import { useForm } from 'react-hook-form';
import "../styles/App.css";
import { login } from "../api/auth";


const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 1024px;
  height: 100vh;
  border: 3px solid black;
  padding: 10px;
`;
const LoginHeader = styled.div`
    width: 95%;
    height: 15%;
    border: 3px solid red;
    margin-bottom: 20px;
    margin-top: 20px;
    h1 {
        text-align: center;
        font-size: 50px;
        font-family: "Black Han Sans", sans-serif;
        font-weight: 400;
        font-style: normal;
        color: red;
    };
`;
const FormWrapper = styled.div`
    width: 95%;
    height: 75%;
    border: 3px solid red;
    display: flex;
    padding: 10px;
`;
const Article = styled.div`
    padding: 20px;
    margin: 5px;
    border: 3px solid red;
    flex-grow: 7;
`;
export default function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [accessToken, setAccessToken] = useState(null);
    const [refreshToken, setRefreshToken] = useState(null);


    const handleLogin = async () => {
        try {
            const response = await login(username, password);
            setAccessToken(response.access);
            setRefreshToken(response.refresh);
            console.log(response);
        } catch (error) {
            console.error('Login failed', error);
        }
    };
    
    return (
        <Wrapper>
            <LoginHeader>
                <h1>ForHero</h1>
            </LoginHeader>
            <FormWrapper>
                <form className="loginForm" onSubmit={handleLogin}>
                    <div className="form-group">
                        <label htmlFor="email">Email</label>
                        <input
                            id="email"
                            type="email"
                            {...register('email', { required: 'Email is required' })}
                        />
                        {errors.email && <p>{errors.email.message}</p>}
                    </div>
                    <div className="form-group">
                        <label htmlFor="password">Password</label>
                        <input
                            id="password"
                            type="password"
                            {...register('password', { required: 'Password is required' })}
                        />
                        {errors.password && <p>{errors.password.message}</p>}
                    </div>
                    <button className="btn" type="submit">Login</button>
                </form>
                <Article>
                    <h1>text header</h1>
                    <p>text</p>
                </Article>
            </FormWrapper>
        </Wrapper>
    )
}