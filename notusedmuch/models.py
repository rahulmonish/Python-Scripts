from pathlib import Path
from django.utils import timezone
from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from picklefield.fields import PickledObjectField
from deepminer.parsing.module_metadata import ExpressionType
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.utils.translation import ugettext_lazy as _


class ModuleStruct(models.Model):
    module_name = models.CharField(max_length=500)
    complexity = models.IntegerField(null=True)

    def __str__(self):
        return self.module_name


class CompilationUnit(models.Model):
    module = models.ForeignKey(ModuleStruct, related_name='compilationUnits', on_delete=models.CASCADE, blank=True,
                               null=True)
    name = models.CharField(max_length=500)
    statementCount = models.IntegerField(default="", blank=True, null=True)
    complexity = models.IntegerField(null=True)
    identifier = models.IntegerField(db_index=True)
    capabilityLabel = GenericRelation(
        'ApplicationCapability',
        content_type_field='unitType',
        object_id_field='unitId',
    )

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['identifier']


class SourcePosition(models.Model):
    startLine = models.IntegerField(blank=True, null=True)
    endLine = models.IntegerField(blank=True, null=True)


class ExecutionUnit(models.Model):
    name = models.CharField(max_length=500, default="", blank=True)
    statementCount = models.IntegerField(default="", blank=True, null=True)
    complexity = models.IntegerField(null=True)
    returnType = models.TextField(blank=True, null=True)
    sourcePosition = models.OneToOneField(SourcePosition, on_delete=models.CASCADE, blank=True, null=True)
    identifier = models.IntegerField(db_index=True)
    compilation = models.ForeignKey(CompilationUnit, related_name='executionUnits', on_delete=models.CASCADE,
                                    blank=True, null=True)
    capabilityLabel = GenericRelation(
        'ApplicationCapability',
        content_type_field='unitType',
        object_id_field='unitId',
    )

    # secondary data, for the purpose of quickly building the call heirarchy json
    module = models.ForeignKey(ModuleStruct, related_name='executionUnits', on_delete=models.CASCADE,
                               blank=True, null=True)

    def flowBlocks(self):
        from .deepmining_core.grakn_write import graknReadTransaction
        with graknReadTransaction() as tx:
            results = tx.query('''match
                $unit isa execution_unit_type, has db_id {};
                (parent: $unit, child: $block) isa parentship;
                $block has db_id $block_id;
            get;'''.format(self.id))
            blocks = [
                ExecutionBlock.objects.get(pk=result.map()['block_id'].value())
                for result in results
            ]
            blocks.sort(key=lambda b: b.identifier)
            return blocks

    def flowEdges(self):
        from .deepmining_core.grakn_write import graknReadTransaction
        with graknReadTransaction() as tx:
            results = tx.query('''match
                $unit isa execution_unit_type, has db_id {};
                (parent: $unit, child: $block) isa parentship;
                (predecessor: $block, successor: $next_block) isa control_flow;
                $block has db_id $block_id;
                $next_block has db_id $next_block_id;
            get;'''.format(self.id))
            edges = [
                {
                    'predecessor': result.map()['block_id'].value(),
                    'successor': result.map()['next_block_id'].value(),
                }
                for result in results
            ]
            return edges

    class Meta:
        ordering = ['identifier']

    def __str__(self):
        if self.compilation:
            return "{}.{}".format(self.compilation, self.name)
        else:
            return self.name


class InputData(models.Model):
    type = models.TextField(blank=True, null=True)
    argumentName = models.TextField(blank=True, null=True)
    initialValueExpression = models.ForeignKey("Expression", related_name='+',
                                               on_delete=models.CASCADE, blank=True, null=True)
    execUnit = models.ForeignKey(ExecutionUnit, related_name='inputData', on_delete=models.CASCADE,
                                 blank=True, null=True)

    @property
    def name(self):
        return self.argumentName

    def __str__(self):
        return "{} {}".format(self.type, self.name)


class ExecutionBlock(models.Model):
    name = models.CharField(max_length=500, null=True)
    identifier = models.IntegerField(db_index=True)
    statementCount = models.IntegerField(default="", blank=True, null=True)
    complexity = models.IntegerField(null=True)
    blockNumber = models.IntegerField(default="", blank=True, null=True)
    entryCondition = models.ForeignKey("Expression", related_name='+', on_delete=models.SET_NULL, blank=True, null=True)
    exitCondition = models.ForeignKey("Expression", related_name='+', on_delete=models.SET_NULL, blank=True, null=True)
    execBlocks = models.ForeignKey("self", on_delete=models.CASCADE, blank=True, null=True, default=None,
                                   related_name='executionBlocks')
    execUnit = models.ForeignKey(ExecutionUnit, related_name='executionBlocks', on_delete=models.CASCADE, blank=True,
                                 null=True)
    expressions_unordered = models.ManyToManyField('Expression', related_name='block_parents',
                                                   through='BlockExpression', through_fields=('parent', 'child'))
    module = models.ForeignKey(ModuleStruct, related_name='executionBlocks', on_delete=models.CASCADE, blank=True,
                               null=True)
    sourcePosition = models.OneToOneField(SourcePosition, on_delete=models.CASCADE, blank=True, null=True)
    capabilityLabel = GenericRelation(
        'ApplicationCapability',
        content_type_field='unitType',
        object_id_field='unitId',
    )

    @property
    def expressions(self):
        return self.expressions_unordered.order_by('parent_block_relations__index')

    def executionUnit(self):
        return self.execUnit or self.execBlocks and self.execBlocks.executionUnit()

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['identifier']


class SymbolEntry(models.Model):
    symbolText = models.CharField(max_length=500)
    initialValueExpression = models.ForeignKey("Expression",
        related_name='+', on_delete=models.CASCADE, blank=True, null=True)
    type = models.CharField(max_length=500, blank=True, null=True)
    compilationUnit = models.ForeignKey(CompilationUnit, related_name='symbolEntries', on_delete=models.CASCADE,
                                        blank=True, null=True)
    executionUnit = models.ForeignKey(ExecutionUnit, related_name='symbolEntries', on_delete=models.CASCADE, blank=True,
                                      null=True)
    executionBlock = models.ForeignKey(ExecutionBlock, related_name='symbolEntries', on_delete=models.CASCADE,
                                       blank=True, null=True)

    @property
    def name(self):
        return self.symbolText

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = 'SymbolEntries'


class Expression(models.Model):
    identifier = models.IntegerField(db_index=True)
    expressionType = models.IntegerField()
    nameString = models.TextField(default="")
    children = models.ManyToManyField('Expression', related_name='parents',
                                      through='ExpressionExpression', through_fields=('parent', 'child'))
    returnExpressionIn = models.ForeignKey(ExecutionUnit, related_name='returnExpressions',
                                           on_delete=models.CASCADE, blank=True, null=True)

    # secondary data, for the purpose of quickly building the call heirarchy json
    module = models.ForeignKey(ModuleStruct, related_name='expressions', on_delete=models.CASCADE, blank=True,
                               null=True)

    # Deepminer computed fields
    variable_symbolEntry = models.ForeignKey(SymbolEntry, related_name='references',
                                             on_delete=models.SET_NULL, null=True, blank=True)
    variable_inputData = models.ForeignKey(InputData, related_name='references',
                                           on_delete=models.SET_NULL, null=True, blank=True)
    calls = models.ManyToManyField(ExecutionUnit, through='Invoke',
                                   through_fields=('expression', 'executionUnit'),
                                   related_name='callers', blank=True)
    resolved = models.NullBooleanField()
    # these next two are for quickly jumping from function or block to it's children in call heirarchy
    resolvedInBlock = models.ForeignKey(ExecutionBlock, related_name='resolvedInvokes',
                                        on_delete=models.CASCADE, blank=True, null=True)
    resolvedInExecutionUnit = models.ForeignKey(ExecutionUnit, related_name='resolvedInvokes',
                                                on_delete=models.CASCADE, blank=True, null=True)

    @property
    def receiver(self):
        try:
            return self.children.get(parent_relations__index=ExpressionExpression.RECEIVER_INDEX)
        except Expression.DoesNotExist:
            return None

    @property
    def arguments(self):
        return self.children.filter(parent_relations__index__gte=0).order_by('parent_relations__index')

    @property
    def variable(self):
        return self.variable_symbolEntry or self.variable_inputData

    @property
    def name(self):
        return self.nameString

    def execution_blocks(self, include_references=False):
        yield from self.block_parents.all()
        for parent in self.parents.all():
            yield from parent.execution_blocks(include_references)
        if include_references:
            yield from ExecutionBlock.objects.filter(entryCondition=self)
            yield from ExecutionBlock.objects.filter(exitCondition=self)
            yield from ExecutionBlock.objects.filter(symbolEntries__initialValueExpression=self)

    def executionBlock(self, include_references=False):
        for block in self.execution_blocks(include_references=include_references):
            return block
        return None

    def execution_units(self, include_references=False):
        for block in self.execution_blocks(include_references):
            yield block.executionUnit()
        if include_references:
            yield from ExecutionUnit.objects.filter(symbolEntries__initialValueExpression=self)
            yield from ExecutionUnit.objects.filter(inputData__initialValueExpression=self)

    def executionUnit(self, include_references=False):
        for unit in self.execution_units(include_references):
            return unit
        return None

    def compilation_units(self, include_references=False):
        for unit in self.execution_units(include_references):
            yield unit.compilation
        if include_references:
            yield from CompilationUnit.objects.filter(symbolEntries__initialValueExpression=self)

    def compilationUnit(self, include_references=False):
        for unit in self.compilation_units(include_references):
            return unit
        return None

    def __str__(self):
        return "{} {}.{}[{}]".format(
            ExpressionType(self.expressionType).name,
            self.executionUnit(include_references=True),
            self.name,
            self.id,
        )

    class Meta:
        ordering = ['identifier']


class ChildExpression(models.Model):
    index = models.SmallIntegerField()

    class Meta:
        abstract = True
        unique_together = ['parent', 'index']
        ordering = ['index']


class BlockExpression(ChildExpression):
    parent = models.ForeignKey(ExecutionBlock, on_delete=models.CASCADE, related_name='expression_relations')
    child = models.ForeignKey(Expression, on_delete=models.CASCADE, related_name='parent_block_relations')


class ExpressionExpression(ChildExpression):
    parent = models.ForeignKey(Expression, on_delete=models.CASCADE, related_name='child_relations')
    child = models.ForeignKey(Expression, on_delete=models.CASCADE, related_name='parent_relations')
    keyword = models.CharField(max_length=500, null=True)

    RECEIVER_INDEX = -1


class Invoke(models.Model):
    expression = models.ForeignKey(Expression, on_delete=models.CASCADE, related_name='invokes')
    executionUnit = models.ForeignKey(ExecutionUnit, on_delete=models.CASCADE, related_name='invokes', null=True)
    dependency = models.ForeignKey('DependencyObject', related_name='expressions',
                                   on_delete=models.CASCADE, null=True, blank=True)
    expression_value = models.OneToOneField(Expression, on_delete=models.CASCADE, null=True)
    # a pattern matching the flow leading to this data operation.
    # currently, a comma-seperated list of expression ids, most recent to least recent
    # Written by deepminer.parsing.flow.FlowSet.serializedPath
    flowTail = models.CharField(max_length=500, blank=False)

    class Meta:
        unique_together = ('expression', 'executionUnit')

    def __str__(self):
        return "{} -> {} via {}".format(self.expression, self.executionUnit, self.dependency)


class Import(models.Model):
    name = models.CharField(max_length=500)
    compilationUnit = models.ForeignKey(CompilationUnit, related_name='imports', on_delete=models.CASCADE, blank=True,
                                        null=True)
    rank = models.IntegerField(db_index=True)

    class Meta:
        ordering = ['rank']


class Tag(models.Model):
    label = models.CharField(max_length=500)
    expression = models.ForeignKey(Expression, related_name='tags', on_delete=models.CASCADE, blank=True, null=True)

    def __str__(self):
        return self.label


class Annotation(models.Model):
    annotation = PickledObjectField(blank=True, null=True)
    compilationUnit = models.ForeignKey(CompilationUnit, related_name='annotations',
                                        on_delete=models.CASCADE, blank=True, null=True)
    executionUnit = models.ForeignKey(ExecutionUnit, related_name='annotations',
                                      on_delete=models.CASCADE, blank=True, null=True)
    symbolEntry = models.ForeignKey(SymbolEntry, related_name='annotations',
                                    on_delete=models.CASCADE, blank=True, null=True)
    expression = models.ForeignKey(Expression, related_name='annotations',
                                   on_delete=models.CASCADE, blank=True, null=True)


class RepositoryType(models.Model):
    name = models.CharField(max_length=500)
    pluginName = models.CharField(max_length=500)
    enabled = models.BooleanField()


class Language(models.Model):
    name = models.CharField(max_length=500)

    def __str__(self):
        return self.name


class TechnologyGroup(models.Model):
    group = models.IntegerField(blank=True, null=True)
    desc = models.TextField(blank=True)
    primaryLanguage = models.ForeignKey(Language, related_name='primaryLanguage', on_delete=models.CASCADE, blank=True,
                                        null=True)
    secondaryLanguages = models.ManyToManyField(Language, related_name='secondaryLanguages', blank=True)

    def __str__(self):
        return "{}[{}]".format(
            self.primaryLanguage.name,
            self.id,
        )


# Modified portfolio implementation
class Portfolio(models.Model):
    portfolioName = models.CharField(max_length=500)

    def __str__(self):
        return self.portfolioName


class Repository(models.Model):
    url = models.CharField(max_length=255)
    checkout_dir_name = models.CharField(max_length=255)

    def __str__(self):
        return self.url

    class Meta:
        verbose_name_plural = "Repository"


class ApplicationModule(models.Model):
    application = models.CharField(max_length=500)
    module = models.CharField(max_length=500)
    sourceFiles = models.IntegerField(blank=True, null=True)
    location = models.CharField(max_length=500)
    repositoryType = models.ForeignKey(RepositoryType, on_delete=models.CASCADE, blank=True, null=True)
    technology = models.ForeignKey(TechnologyGroup, on_delete=models.CASCADE, blank=True, null=True)
    portfolio = models.ForeignKey(Portfolio, related_name='applications',
                                  on_delete=models.CASCADE)  # Modified portfolio implementation
    moduleStruct = models.OneToOneField(ModuleStruct, related_name='application',
                                        on_delete=models.CASCADE, null=True, blank=True)
    updateDate = models.DateTimeField(default=timezone.now, blank=True)
    repository = models.ForeignKey(Repository, related_name="application_modules", on_delete=models.CASCADE)

    def __str__(self):
        return "{} {}".format(self.application, self.module)

    def allDependencies(self):
        """ build dependencies and runtime dependencies both """
        for buildConfiguration in self.buildData.all():
            yield from buildConfiguration.buildDependencies.all()
        for runInstance in self.runInstances.all():
            yield from runInstance.runtimeDependencies.all()

    class Meta:
        permissions = (("view_all_applications", "To view all the applications"),
                       ("view_all_portfolio_applications", "To view all the applications in a portfolio"),
                       ("view_required_applications","To view selected applications"))


class SourceFileList(models.Model):
  fileName = models.CharField(max_length=500)
  module = models.ForeignKey(ApplicationModule, related_name='fileList', on_delete=models.CASCADE, blank=True,
                             null=True)


class Middleware(models.Model):
    name = models.CharField(max_length=500)
    version = models.CharField(max_length=500)


class Platform(models.Model):
    os = models.CharField(max_length=500)
    version = models.CharField(max_length=500)
    description = models.CharField(max_length=500)
    middleware = models.ManyToManyField(Middleware, related_name='middleware', blank=True)


class ProductionFormat(models.Model):
    Format = models.IntegerField(default=None, blank=True)


class DependencyObject(models.Model):
    portfolio = models.ForeignKey(Portfolio, related_name='dependencies', on_delete=models.CASCADE)
    dependencyName = models.TextField(default="", blank=True)
    version = models.TextField(default=None, blank=True)
    type = models.TextField(default=None, blank=True)
    resolved_module = models.ForeignKey(ApplicationModule, related_name='+', on_delete=models.CASCADE, null=True)

    def __str__(self):
        return "{} {}".format(self.dependencyName, self.version)


class BuildData(models.Model):
    module = models.ForeignKey(ApplicationModule, related_name='buildData', on_delete=models.CASCADE, blank=True,
                               null=True)
    buildConfigName = models.CharField(max_length=500, null=True, blank=True)
    targetPlatform = models.ForeignKey(Platform, on_delete=models.CASCADE, blank=True, null=True)
    format = models.ForeignKey(ProductionFormat, on_delete=models.CASCADE, blank=True, null=True)
    buildDependencies = models.ManyToManyField(DependencyObject, related_name='buildDependencies', blank=True)


class Infrastructure(models.Model):
    infra_id = models.AutoField(primary_key=True)
    boxName = models.CharField(max_length=500)
    boxType = models.CharField(max_length=500)
    platform = models.CharField(max_length=500)
    portfolio = models.ForeignKey(Portfolio, related_name='infrastructures',
                    on_delete=models.CASCADE)  # Modified portfolio implementation
    # FIXME: Should this ManyToManyField route thru the RunInstances table, instead of being seperate?
    # Are there any infrastructures <--> ApplicationModule pairing that aren't RunInstances?
    applications = models.ManyToManyField(ApplicationModule,
                                          # through='RunInstance', # Proposed,
                                          related_name='infrastructures')

    def __str__(self):
        return "{} ({}-{})".format(self.boxName, self.boxType, self.platform)


class RunInstance(models.Model):
    module = models.ForeignKey(ApplicationModule, related_name='runInstances', on_delete=models.CASCADE)
    classification = models.CharField(max_length=500, null=True, blank=True)
    boxDetails = models.ForeignKey(Infrastructure, on_delete=models.CASCADE)
    build = models.ForeignKey(BuildData, on_delete=models.CASCADE, blank=True, null=True)
    runtimeDependencies = models.ManyToManyField(DependencyObject, related_name='runtimeDependencies', blank=True)

    def __str__(self):
        return '{} (build {})'.format(self.module, self.classification)

    class Meta:
        verbose_name_plural = 'RunInstances'


class DataDependencyEntity(models.Model):
    sourceType = models.CharField(max_length=500, null=True, blank=True)  # this field indicates type of the data source
    infrastructure = models.CharField(max_length=500, null=True, blank=True)  # hostname or machine name or server name
    port = models.IntegerField(null=True, blank=True)  # port number
    system = models.CharField(max_length=500, null=True, blank=True)  # database or path or context path
    entity = models.CharField(max_length=500, null=True,
                              blank=True)  # table or filename or resource(REST) or operation(SOAP)
    runInstance = models.ManyToManyField(RunInstance, related_name='dataDependencies', blank=True)
    portfolio = models.ForeignKey(Portfolio, related_name='datadependencies',
                                  on_delete=models.CASCADE)  # Modified portfolio implementation

    def __str__(self):
        return '{} {}{}.{}.{}'.format(
            self.sourceType,
            self.infrastructure,
            ':{}'.format(self.port) if self.port else '',
            self.system,
            self.entity,
        )

    class Meta:
        verbose_name_plural = 'DataDependencyEntities'
        unique_together = (
            'sourceType',
            'infrastructure',
            'port',
            'system',
            'entity',
            'portfolio',
        )


class ApplicationInternalFlow(models.Model):
    module = models.ForeignKey(ApplicationModule, related_name='flows',
                               on_delete=models.CASCADE)
    entryPoint = models.ForeignKey(ExecutionUnit, related_name='+',
                                   on_delete=models.CASCADE)

    def __str__(self):
        return '{} entryPoint {}'.format(self.module, self.entryPoint)


class FlowExecutionUnit(models.Model):
    START = 'S'
    CALL = 'C'
    RETURN = 'R'
    TYPE_CHOICES = [
        (START, "Start at"),
        (CALL, "Call"),
        (RETURN, "Return to"),
    ]
    flow = models.ForeignKey(ApplicationInternalFlow, related_name='units',
                             on_delete=models.CASCADE)
    unit = models.ForeignKey(ExecutionUnit, related_name='+',
                             on_delete=models.CASCADE)
    type = models.CharField(max_length=1, choices=TYPE_CHOICES)
    first_expression = models.ForeignKey(Expression, related_name='+',
                                         on_delete=models.CASCADE, null=True)
    last_expression = models.ForeignKey(Expression, related_name='+',
                                        on_delete=models.CASCADE, null=True)

    class Meta:
        ordering = ['id']


class FlowExecutionBlock(models.Model):
    unit = models.ForeignKey(FlowExecutionUnit, related_name='blocks', on_delete=models.CASCADE)
    block = models.ForeignKey(ExecutionBlock, related_name='+', on_delete=models.CASCADE)

    class Meta:
        ordering = ['id']


class DataField(models.Model):
    fieldType = models.CharField(max_length=500, blank=False)
    fieldName = models.CharField(max_length=500, blank=False)
    dataEntity = models.ForeignKey(DataDependencyEntity, related_name='fields',
                                   on_delete=models.CASCADE, blank=True, null=True)

    class Meta:
        unique_together = ('fieldName', 'dataEntity')

    def __str__(self):
        return '{}.{} ({})'.format(self.dataEntity, self.fieldName, self.fieldType)


class DataOperation(models.Model):
    flow_blocks = models.ManyToManyField(FlowExecutionBlock, related_name="operations")
    readOrWrite = models.BooleanField(blank=False)
    dataDetails = models.ForeignKey(DataDependencyEntity, related_name="operations",
                                    on_delete=models.CASCADE, blank=True, null=True)
    fields = models.ManyToManyField(DataField, related_name="operations", blank=False)
    externalInvoke = models.ForeignKey(Expression, related_name="operations",
                                       on_delete=models.CASCADE, blank=False)
    expression_value = models.OneToOneField(Expression, on_delete=models.CASCADE, null=True)
    # a pattern matching the flow leading to this data operation.
    # currently, a comma-seperated list of expression ids, most recent to least recent
    # Written by deepminer.parsing.flow.FlowSet.serializedPath
    flowTail = models.CharField(max_length=500, blank=False)

    class Meta:
        unique_together = ('readOrWrite', 'dataDetails', 'externalInvoke', 'flowTail')

    def __str__(self):
        return '{} {} {}'.format(
            self.externalInvoke.name,
            "writing" if self.readOrWrite else "reading",
            self.dataDetails,
        )


class ExecutionDetails(models.Model):
    flowId = models.CharField(max_length=500, null=True, blank=True)
    predecessor = models.ForeignKey(ApplicationModule,related_name="predecessor_application",on_delete=models.CASCADE, blank=True, null=True)
    successor = models.ForeignKey(ApplicationModule,related_name="successor_application",on_delete=models.CASCADE, blank=True, null=True)
    runInstances = models.ForeignKey(RunInstance, related_name='executionDetails', on_delete=models.CASCADE,
                                     blank=True, null=True)

    def __str__(self):
        return '{} (exec {})'.format(self.runInstances, self.id)

    class Meta:
        verbose_name_plural = 'ExecutionDetails'


class CapabilityStruct(models.Model):
    label = models.CharField(max_length=500, unique=True, blank=True, null=True)
    parent = models.ForeignKey("self", related_name='children',
                               on_delete=models.CASCADE, blank=True, null=True)
    portfolio = models.ForeignKey(Portfolio, related_name='capabilities',
                                  on_delete=models.CASCADE)  # Modified portfolio implementation
    modules = models.ManyToManyField(
        ApplicationModule,
        through='ApplicationCapability',
        related_name='capabilityLabel',
    )

    @property
    def parent_list(self):
        """ serialization expects parent to be an array for some reason """
        return [self.parent] if self.parent else []

    def __str__(self):
        return "{}[{}]".format(self.label, self.id)


class CsvFileRead(models.Model):
    csv_file = models.FileField(upload_to='csvfile/')


class ApplicationCapability(models.Model):
    module = models.ForeignKey(ApplicationModule, related_name='applicationCapability', on_delete=models.CASCADE)
    capability = models.ForeignKey(CapabilityStruct, related_name='app_capability', on_delete=models.CASCADE)
    unitId = models.IntegerField(blank=True, null=True)
    unitType = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    unit = GenericForeignKey('unitType', 'unitId')
    FIELD_CHOICES = (('Actual', 'Actual'), ('Predicted', 'Predicted'), ('User_Confirmed', 'User_Confirmed'),
                     ('User_Rejected', 'User_Rejected'))
    capability_choice = models.CharField(max_length=500, choices=FIELD_CHOICES, blank=True)
    class Meta:
        verbose_name_plural = 'ApplicationCapabilities'

    def __str__(self):
        return '"{}" module: {}; unit: {!r}'.format(
            self.capability, self.module, self.unit)


class ConfigModule(models.Model):
    config_data = PickledObjectField(blank=True, null=True)
    application = models.ForeignKey(ApplicationModule, related_name='nonSourceFiles',
                                    on_delete=models.CASCADE, null=True, blank=True)

    @property
    def path(self):
        return Path(self.config_data['filename'])

    def __str__(self):
        return self.path.name


class SearchField(models.Model):
    FIELD_CHOICES = (
        ('Applications','Applications'),
        ('Infrastructure','Infrastructure'),
        ('Capability','Capability'),
        ('Attributes','Attributes'),
        )
    search_field = models.CharField(max_length = 500,choices = FIELD_CHOICES)

    def __str__(self):
        return self.search_field


class SearchInfo(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,on_delete = models.CASCADE)
    search_key = models.CharField(max_length=500)
    search_date = models.DateTimeField(auto_now_add=True)
    search_field = models.ManyToManyField(SearchField, related_name='field', blank=True)


class Attribute(models.Model):
    application = models.ForeignKey(ApplicationModule, on_delete=models.CASCADE)
    entityType = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    entityId = models.PositiveIntegerField()
    entity = GenericForeignKey('entityType', 'entityId')
    attributeKey = models.CharField(max_length=500, db_index=True)
    attributeLabel = models.CharField(max_length=500, db_index=True)
    attributeValue = models.CharField(max_length=500, db_index=True)

    class Meta:
        unique_together = (
            'application',
            'entityType',
            'entityId',
            'attributeKey',
        )

    def __str__(self):
        return "{}({}): {}".format(self.attributeLabel, self.attributeKey, self.attributeValue)


class User(AbstractUser):
    is_socialuser_active = models.BooleanField(_('active social user'), default=False)
    class Meta:
        permissions = (("Executive", "To access Executive view"),
                       ("Analyst", "To access Analyst view"),
                       ("Admin", "For Administrative access"))
