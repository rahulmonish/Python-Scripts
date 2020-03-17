import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import django
os.environ['DJANGO_SETTINGS_MODULE'] = 'dm_app.settings.development'
django.setup()

from dm_restAPI import models

columns = [
    'Id',
    'Identifier',
    'Expression Type',
    'Name String',
    'Receiver ID',
    'Parent ID',
    'Module Id',
    'Module Name',
    'Module Complexity',
    'Data Operation',
    'Database',
    'Execution Block ID',
    'Execution Block Identifier',
    'Execution Block Name',
    'Execution Block Block ID',
    'Execution Block Complexity',
    'Execution Block Entry Condition',
    'Execution Block Exit Condition',
    'Execution Block Start Line',
    'Execution Block End Line',
    'Execution Unit ID',
    'Execution Unit Identifier',
    'Execution Unit Name',
    'Execution Unit Return Type',
    'Execution Unit Start Line',
    'Execution Unit End Line',
    'Compilation Unit ID',
    'Compilation Unit Identifier',
    'Compilation Unit Name',
    'Compilation Unit Complexity',
    'Compilation Unit Statement Count',
]

expression = (
    models.Expression.objects.all().order_by('id')
)
row = []
with open('expression_clustter.csv', 'w',newline='') as writeFile:
    writer = csv.DictWriter(writeFile,fieldnames=columns)
    writer.writeheader()
    for exp in expression:
        exp_dict =     {
            'Id': exp.id,
            'Identifier' : exp.identifier,
            'Expression Type' : exp.expressionType,
            'Name String' : exp.nameString,
            }
        if exp.receiver:
            exp_dict['Receiver ID']=exp.receiver.id
        # else:
        #     row.append(None)
        if exp.parent_id:
            exp_dict['Parent ID']=exp.parent_id
        # else:
        #     row.append(None)

        if exp.module:
            module = exp.module
            exp_dict['Module Id']=module.id
            exp_dict['Module Name']=module.module_name
            exp_dict['Module Complexity']=module.complexity

        if exp.operations.all():
            operation = exp.operations.all()[0]
            exp_dict['Data Operation']=operation.readOrWrite
            exp_dict['Database']=operation.dataDetails.entity
        if exp.execBlock:
            block = exp.execBlock
            exp_dict['Execution Block ID']=block.id
            exp_dict['Execution Block Identifier']=block.identifier
            exp_dict['Execution Block Name']=block.name
            exp_dict['Execution Block Block ID']=block.blockNumber
            exp_dict['Execution Block Complexity']=block.complexity
            exp_dict['Execution Block Entry Condition']=block.entryCondition.id
            exp_dict['Execution Block Exit Condition']=block.exitCondition.id
            exp_dict['Execution Block Start Line']=block.sourcePosition.startLine
            exp_dict['Execution Block End Line']=block.sourcePosition.endLine
            if block.execUnit:
                execution = block.execUnit
                exp_dict['Execution Unit ID']=execution.id
                exp_dict['Execution Unit Identifier']=execution.identifier
                exp_dict['Execution Unit Name']=execution.name
                exp_dict['Execution Unit Return Type']=execution.returnType
                exp_dict['Execution Unit Start Line']=execution.sourcePosition.startLine
                exp_dict['Execution Unit End Line']=execution.sourcePosition.endLine
                if execution.compilation:
                    compilation = execution.compilation
                    exp_dict['Compilation Unit ID']=compilation.id
                    exp_dict['Compilation Unit Identifier']=compilation.identifier
                    exp_dict['Compilation Unit Name']=compilation.name
                    exp_dict['Compilation Unit Complexity']=compilation.complexity
                    exp_dict['Compilation Unit Statement Count']=compilation.statementCount
        row.append(exp_dict)
    writer.writerows(row)
writeFile.close()