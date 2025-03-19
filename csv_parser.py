import csv
import os
import chardet

file_name = r"C:\Users\L03140374\buyer_persona_analysis\data\Viña Real - Analisis de Ventas UTF-8.csv"

#Método Parser, obetivo es poder reutilizar el codigo con otros archivos tambien. 
def parser (path):

    '''
    Hace la revisión de un arhcivo CSV y regresa la metadata del archivo. 

    Parametro path = dirección del archivo csv. 

    Resultado = Metadata, información del archivo y data que está dentro del el. 
    '''
    #Diccionario que almacena los resultados del método. 
    results = {
        'name': "",
        'success': False,
        'message': '',
        'encoding': None,
        'headers': [],
        'row_count': 0,
        'column_count': 0,
        'Potential_issues': []
    }
    #Revisa si el path es valido y regresa la excepción para mostrar si el archivo no existe. 
    if not os.path.exists(path):
        results['success'] = False
        results['message'] = f'Error: file does not exist'
        return results
    
    #Revisa si el archivo es un CSV. 
    if not path.lower().endswith('.csv'):
        results['success'] = False
        results['message'] = f'Error: file is not CSV File'
        return results
    

    #Detectar que tipo de "encoding" es el archivo, se salva en el diccionario para reutilizarlo al momento de usar el with-open del archivo. 
    try: 
        with open (path, 'rb') as f: 
            data = f.read(10000)
        encoding_result = chardet.detect(data)
        results['encoding'] = encoding_result['encoding']
    except PermissionError as e: 
        print("Hay un error en el tipo de permisos: {e}")
    except FileNotFoundError as e: 
        print("Hay un error porque este archivo no existe: {e}")

    #Obtener el nombre de la variable. 
        temporal_split_name_path = os.path.split(path)
        results["name"] = temporal_split_name_path[-1]

    #Bloque try - except error sobre 
    try:
    #Primera pasada oara obtener Número de columnas, encabezados y números de filas. 
        with open (path, 'r', encoding=results['encoding']) as csv_file:
            csvreader = csv.reader(csv_file) #Reader for the CSV    
            #Guarda los nombres y cuenta los encabezados en la lista "Headers", que están dentro del archivo que se está corriendo.
            temporal_header_list = next(csvreader)
            results['column_count'] = len(temporal_header_list)
            results['headers'] = temporal_header_list
            results['row_count'] = sum(1 for row in csvreader)
            #Segunda pasada para obtener posibles errores dentro de las filas 
            
        with open (path, 'r', encoding=results['encoding']) as csv_file:
            csvreader = csv.reader(csv_file) #Reader for the CSV
            next(csvreader) #Salta los encabezador
            #Cuenta el numero de filas que están dentro del archivo .csv
            for i, row in enumerate(csvreader):
                if len(row) != results['column_count']:
                    results['Potential_issues'].append(
                        f"Row {i+1}: Expected {results['column_count']} columns, found {len(row)}"
                        )
                
                    for j, cell in enumerate(row):
                        if not cell.strip():
                            results['Potential_issues'].append(
                            f"Row {i+1}, Column '{results['headers'][j]}': Empty value"
                            )

        #Posibles errores
    except FileNotFoundError as e: 
        print(f"A specific error ocurred: {e}")
    except TypeError as e: 
        print(f"El error persiste en la sintax: {e}")
    
    if len(results['Potential_issues']) == 0:
        results['message'] = "Successfully parsed with no issues"
    else:
        results['message'] = f"Successfully parsed with {len(results['Potential_issues'])} potential issues"
    results['success'] = True
        
    return results

def print_summary(results):
    """
    Imprime un resumen formateado de los resultados del análisis CSV.
    
    Args:
        results (dict): Diccionario con los resultados del parser
    """
    # Definir colores para la terminal (opcional)
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Imprimir encabezado principal
    print("\n" + "="*80)
    print(f"{BOLD}RESULTADOS DEL ANÁLISIS CSV{END}".center(80))
    print("="*80 + "\n")
    
    # Sección 1: Información general del archivo
    print(f"{BOLD}{UNDERLINE}INFORMACIÓN GENERAL:{END}")
    print(f"  • Archivo: {BLUE}{results['name']}{END}")
    print(f"  • Estado: {GREEN if results['success'] else RED}{results['message']}{END}")
    print(f"  • Codificación detectada: {results['encoding']}")
    print()
    
    # Sección 2: Estructura del archivo
    print(f"{BOLD}{UNDERLINE}ESTRUCTURA DEL ARCHIVO:{END}")
    print(f"  • Número de filas: {results['row_count']}")
    print(f"  • Número de columnas: {results['column_count']}")
    print(f"  • Encabezados: {', '.join(results['headers'][:5])}...")
    print()
    
    # Sección 3: Análisis de problemas
    print(f"{BOLD}{UNDERLINE}ANÁLISIS DE PROBLEMAS:{END}")
    
    if not results['Potential_issues']:
        print(f"  {GREEN}¡No se encontraron problemas en el archivo!{END}")
    else:
        # Contar problemas por columna
        issues_by_column = {}
        total_issues = len(results['Potential_issues'])
        
        for issue in results['Potential_issues']:
            # Extraer nombre de columna del mensaje
            if "Column '" in issue:
                column_name = issue.split("Column '")[1].split("'")[0]
                issues_by_column[column_name] = issues_by_column.get(column_name, 0) + 1
        
        # Mostrar resumen por columna
        print(f"  • Total de problemas encontrados: {YELLOW}{total_issues}{END}")
        print(f"  • Resumen por columna:")
        
        # Ordenar columnas por número de problemas (de mayor a menor)
        sorted_columns = sorted(issues_by_column.items(), key=lambda x: x[1], reverse=True)
        
        for column, count in sorted_columns:
            percentage = (count / total_issues) * 100
            print(f"    - {YELLOW}{column}{END}: {count} problemas ({percentage:.1f}%)")
        
        # Mostrar ejemplos de problemas
        print(f"\n  • Ejemplos de problemas encontrados:")
        for issue in results['Potential_issues'][:5]:  # Mostrar solo los primeros 5
            print(f"    - {issue}")
        
        if len(results['Potential_issues']) > 5:
            print(f"    - ... y {len(results['Potential_issues']) - 5} problemas más")
    
    # Línea final
    print("\n" + "="*80)

result = parser(file_name)
print_summary(result)