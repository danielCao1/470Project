import React, { Fragment } from "react";
 
import {
    Button,
    Alert,
    Card,
    CardImg,
    CardBody,
    CardTitle,
    CardSubtitle,
    CardText,
} from "react-bootstrap";
 
const BANNER = "https://i.imgur.com/CaKdFMq.jpg";
 
const LeftCard = () => (
    <Fragment>
        <Alert color="danger" className="d-none d-lg-block">
            <strong>Account not activated.</strong>
        </Alert>
 
        <Card>
            <CardImg
                top
                width="100%"
                src={BANNER}
                alt="banner"
            />
            <CardBody>
                <CardTitle
                    className="h3 mb-2 pt-2 
                    font-weight-bold text-secondary"
                >
                    ReactJS Tutorials
                </CardTitle>
                <CardSubtitle
                    className="text-secondary mb-3 
                        font-weight-light text-uppercase"
                    style={{ fontSize: "0.8rem" }}
                >
                    ReactJS Tutorials, GFG
                </CardSubtitle>
                <CardText
                    className="text-secondary mb-4"
                    style={{ fontSize: "0.75rem" }}
                >
                    ReactJS is a declarative, efficient, and
                    flexible JavaScript library for building
                    user interfaces.
                </CardText>
                <Button
                    color="success"
                    className="font-weight-bold"
                >
                    Start Learning
                </Button>
            </CardBody>
        </Card>
    </Fragment>
);
 
export default LeftCard;