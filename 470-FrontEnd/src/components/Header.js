import React from "react";
import logo from "../logo.svg";
 
import {
    Container,
    Row,
    Col,
    Form,
    Input,
    Button,
    Navbar,
    Nav,
    NavbarBrand,
    NavLink,
    NavItem,
    Dropdown,
    DropdownToggle,
    DropdownMenu,
    DropdownItem,
} from "react-bootstrap";
 
const AVATAR =
"https://www.gravatar.com/avatar/429e504af19fc3e1cfa5c4326ef3394c?s=240&d=mm&r=pg";
 
const Header = () => (
    <header>
        <Navbar
            fixed="top"
            color="light"
            light
            expand="xs"
            className="border-bottom border-gray bg-white"
            style={{ height: 80 }}
        >
            <Container>
                <Row
                    noGutters
                    className="position-relative w-100 
                            align-items-center display-flex flex-row"
                >
                    <Col className="d-none d-lg-flex 
                            justify-content-start">
                        <Nav
                            className="mrx-auto 
                                    display-flex flex-row"
                            navbar
                        >
                            <NavItem className="d-flex 
                                    align-items-center mx-5">
                                <NavLink
                                    className="font-weight-bold"
                                    href="/"
                                >
                                    <img
                                        src={AVATAR}
                                        alt="avatar"
                                        className="img-fluid 
                                                rounded-circle"
                                        style={{
                                            width: 36,
                                        }}
                                    />
                                </NavLink>
                            </NavItem>
 
                            <NavItem className="d-flex 
                                    align-items-center mx-5">
                                <NavLink
                                    className="font-weight-bold"
                                    href="/"
                                >
                                    Home
                                </NavLink>
                            </NavItem>
 
                            <NavItem className="d-flex 
                                    align-items-center mx-5">
                                <NavLink
                                    className="font-weight-bold"
                                    href="/"
                                >
                                    Electronics
                                </NavLink>
                            </NavItem>
                        </Nav>
                        <Dropdown
                            className="d-flex 
                                    align-items-center overflow-x"
                            nav
                            inNavbar
                        >
                            <DropdownToggle
                                className="font-weight-bold"
                                nav
                                caret
                            >
                                fashion
                            </DropdownToggle>
                            <DropdownMenu right>
                                <DropdownItem
                                    className="font-weight-bold 
                                            text-secondary text-uppercase"
                                    header
                                    disabled
                                >
                                    Learn React
                                </DropdownItem>
                                <DropdownItem divider />
                                <DropdownItem>
                                    Men
                                </DropdownItem>
                                <DropdownItem>
                                    Women
                                </DropdownItem>
                                <DropdownItem>
                                    Baby and Kids
                                </DropdownItem>
                            </DropdownMenu>
                        </Dropdown>
                    </Col>
 
                    <Col
                        className="d-flex justify-content-xs-start 
                                    justify-content-lg-center"
                    >
                        <NavbarBrand
                            className="d-inline-block p-0"
                            href="/"
                            style={{ width: 80 }}
                        >
                            <img
                                src={logo}
                                alt="logo"
                                className="position-relative img-fluid"
                            />
                        </NavbarBrand>
                    </Col>
                </Row>
            </Container>
        </Navbar>
    </header>
);
 
export default Header;