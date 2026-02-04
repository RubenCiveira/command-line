quiero continuar con @command-line la idea es disponer de un agente multi capacidades que sea capaz de detectar inteción via líena de comandos, y que sepa determinar que es lo que se está pidiendo. Cosas que me son importantes:

    - que pueda usar skills o agentes específicos para tareas concretas (si añado un fichero con una skill sobre conventional commit, o sobre versionado semantico que sea capaz de saber que debe cargar esa skill para esa tarea). 
    
    - que sepa catalogar los conocimientos necesarios para la tarea, y así poder buscar en un rag información adecuada. 
    
    - que sepa obtener contexto del espacio de trabajo del usuario (si se pide un "haz una pr para shell-back" que sepa que hay una carpeta con un proyecto "sgt-shellapi" a la que seguramente esté haciendo referencia. 
    
    - que sepa reconocer directorios con documentos de conocimiento y que pueda catalogarls e indexarlos en un rag de forma progresiva como tarea de fondo. 
    
    - que tenga tools y mcps para interactuar. 
    
    - que pueda intentar pedir más información al usuario usando json-schema para lanzarle preguntas (pero que sea capaz de reaccionar si el usuario no proporciona la respuesta). 
    
Quiero el sistema extensible a base a interfaces para poder cambiar fácilmente el espacio de trabajo del usuario que son directorios actualmente por algo distinto o cloud en el futuro.  Genera el plan detallado, por issues y detallando un "fichero a fichero" para desarrollar la ide.
