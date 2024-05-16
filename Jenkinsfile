pipeline {
    agent any
    environment {
        JENKINS_HOME = "$JENKINS_HOME"
        BUILD = "${JENKINS_HOME}/workspace/style_recognition1"
        DOCKER_IMAGE_NAME = 'style_recognition1'
    }

    stages{
        
        stage('Build Docker image'){
            steps {
                sh 'docker build -t ${DOCKER_IMAGE_NAME} .'
            }
        }
        
        stage( 'RUN Docker'){
            steps{
                sh 'docker run -d -p 8501:8501 --name style_recognition-app1 ${DOCKER_IMAGE_NAME}'
            }
        }

        stage( 'Install pytest'){
            steps{
                sh 'pip install pytest'
                sh 'pip install streamlit'
            }
        }
        
        stage( 'RUN Test'){
            steps{
                sh 'python pytest test_main.py'
            }
        }
    }
}
