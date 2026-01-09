# Steup

### venv
Ao trabalhar com python precisamos criar e ativar o "venv", que serve para lidar com diferentes versões do python em diferentes projetos, pois existe o python global e existe o python de cada projeto que é virtualizado.

#### Para criar o venv
> python3 -m venv venv

#### Para ativar
> source ./venv/bin/activate

### requirements
Ao trabalhar com python temos um arquivo de dependencias do projeto, semelhante ao package.json ou project.clj, mas aqui chama-se requirements.txt

#### obs: "pip" é o gerenciador de pacote, por onde instalamos as dependencias, semelhante ao npm.

#### Para instalar Langchain
> pip install langchain

#### Para instalar um facilitador de comunicação com modelos da openai
> pip install langchain-openai

#### Para instalar um facilitador de comunicação com modelos da google
> pip install langchain-google-genai

#### Para carregar as variaveis de ambiente
> pip install python-dotenv

#### Para WebScrapper
> pip install beautifulsoup4

#### Para lidar com arquivos PDF
> pip install pypdf

#### Para lidar com libs da comunidade
> pip install langchain-community

#### Para instalar o PGVector
> pip install langchain-postgres

#### obs: Após baixar todas as dependencias precisamos registra-las no arquivo de requirement.
> pip freeze > requirements.txt

# Tokens
Agora que configurações o setup precisamos obter as api keys da openai e google e adiciona-lás no .env

