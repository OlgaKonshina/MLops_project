pipeline{
    agent any
    enviroment {
    JENKINS_HOME = "$JENKINS_HOME"
    BUILD = "${JENKINS_HOME}/workspace/style_recognition"
    }

    stages{
        stage('Checkout'){
            steps{ 'https://github.com/OlgaKonshina/MLops_project.git'
             }
        }
        stage('Build Docker image'){
            steps {
                sh 'docker build -t style_recognition-img .'
            }
        }
        stage( 'RUN Docker'){
            steps{
                sh 'docker run -d -p 8000:8000 --name style_recognition-app style_recognition-img'
            }
        }
    }
}
