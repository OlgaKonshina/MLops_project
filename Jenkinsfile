pipeline {
    agent any
    options {
        // This is required if you want to clean before build
        skipDefaultCheckout(true)
    }
    environment {
        JENKINS_HOME = "$JENKINS_HOME"
        BUILD = "${JENKINS_HOME}/workspace/style_recognition1"
        DOCKER_IMAGE_NAME = 'style_recognition'
    }

    stages{
        stage('Stop olg container'){
            steps {
                sh 'docker stop ${DOCKER_IMAGE_NAME} && docker rm ${DOCKER_IMAGE_NAME} || true'
                sh 'docker rmi  ${DOCKER_IMAGE_NAME} || true'
            }
        }
        stage('CleanWS'){
            steps {
                cleanWs()
                checkout scm
            }
        }
        stage('Build Docker image'){
            steps {
                sh 'docker build -t ${DOCKER_IMAGE_NAME} .'
            }
        }
        stage( 'RUN Docker'){
            steps{
                sh 'docker run -d -p 8501:8501 --name style_recognition1-app ${DOCKER_IMAGE_NAME}'
            }
        }
         stage( 'Install pytest'){
            steps{
                sh 'pip install pytest'
            }
        }
        stage( 'RUN Test'){
            steps{
                sh 'python3 -m pytest'
            }
        }
    }
}
