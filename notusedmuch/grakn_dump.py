from grakn.client import GraknClient
caller = []
callee = []
block=[]
count=0
with GraknClient(uri="172.16.253.250:48555") as client:
    with client.session(keyspace="dm_graph") as session:
        with session.transaction().read() as read_transaction:
            #results = read_transaction.query("match $unit isa execution_unit_type, has db_id $db_id; (parent: $unit, child: $block) isa parentship; (predecessor: $block, successor: $next_block) isa control_flow; $block has db_id $block_id; $next_block has db_id $next_block_id;get;")
            #results = read_transaction.query("match $unit isa execution_unit_type, has db_id $db_id; (callee: $unit, caller: $next_unit) isa invoke; $callee has db_id $unit_id; $caller has db_id $next_unit_id;get;")
            #results = read_transaction.query("match $unit isa compilation_unit, has db_id $db_id; (parent: $unit, child: $block) isa ancestry; (predecessor: $block, successor: $next_block) isa control_flow; $block has db_id $block_id; $next_block has db_id $next_block_id; get;")
            results = read_transaction.query("match (callee: $callee, caller: $caller, block: $block) isa invoke; $callee has db_id $callee_id; $caller has db_id $caller_id; $block has db_id $block_id; get;")
            #results = read_transaction.query("match (parent: $parent, child: $child) isa parentship; $parent has db_id $parent_id; $child has db_id $child_id; get;")
            #results = read_transaction.query("match $parent plays parent; $child plays child; $child isa compilation_unit; $parent isa compilation_unit; $child has db_id $child_id; $parent has db_id $parent_id; get;")
            #results = read_transaction.query("match $parent isa compilation_unit; $child isa compilation_unit; (parent: $parent, child: $child) isa ancestry; $child has db_id $child_id; $parent has db_id $parent_id; get;")
            
            #persons = results.collect_concepts()
                
            for result in results:
                x= result.map()['callee_id'].value()
                y= result.map()['caller_id'].value()
                z= result.map()['block_id'].value()
                callee.append(x)
                caller.append(y)
                block.append(z)
                print(y,x)
                 
import pickle                
pickle_out = open("callee.pickle","wb")
pickle.dump(callee, pickle_out)
pickle_out.close()

pickle_out = open("block.pickle","wb")
pickle.dump(block, pickle_out)
pickle_out.close()

pickle_out = open("caller.pickle","wb")
pickle.dump(caller, pickle_out)
pickle_out.close()
