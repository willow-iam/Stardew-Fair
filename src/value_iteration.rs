use rurel::mdp::State;
use std::{collections::HashMap};


pub trait StatespaceIterator<S : State>{
    /// The expected value the given state
    fn value(&self,state:S) -> f64;

    fn best_action(&self, state:&S, action_results : fn(&S, &S::A) -> Vec<(f64,S)>) -> S::A;

    /// Update the internal state, which should make the values converge
    fn iterate(&mut self, action_results : fn(&S, &S::A) -> Vec<(f64,S)>);
}


pub struct ValueIterator<S:State>{
    pub state_space:Vec<S>,
    pub values:HashMap<S,f64>
}

fn action_results_value<S:State>(results : Vec<(f64,S)>,values:&HashMap<S,f64>)->f64{
    let rax = results.iter().map(|(p,state)|
        *p * values.get(state).unwrap_or(&0.0)
    ).fold(0.0,|a,b|a+b);

    return rax;
}

impl<S:State> StatespaceIterator<S> for ValueIterator<S>{
    fn value(
        &self,
        state:S
    ) -> f64 {
        *self.values.get(&state).unwrap()
    }

    fn best_action(&self, state:&S, action_results : fn(&S, &S::A) -> Vec<(f64,S)>) -> <S as State>::A {
        state.clone().actions().iter().max_by_key(
            |&action|
            action_results_value(action_results(&state,&action), &self.values).to_bits()
        ).unwrap().clone()
    }

    fn iterate(&mut self, action_results : fn(&S, &S::A) -> Vec<(f64,S)>){
        let mut new_values:HashMap<S, f64> =HashMap::new();
        let mut i=0;
        for state in &self.state_space{
            let state_clone=state.clone();
            if(&state).actions().is_empty() {
                new_values.insert(state_clone,*self.values.get(state).unwrap());
            }
            else{
                let best_value = state.actions().iter().map(
                    |action|
                    action_results_value(action_results(&state,&action), &self.values).to_bits()
                ).max().unwrap();
                //converts to and from bits becuase rust doesn't know how to compare floats.
                new_values.insert(state_clone,f64::from_bits(best_value));
            }
            i=i+1;
            if i%100000==0{
                println!("\tChecked {i} states");
            }
        }
        self.values=new_values.clone();
    }
}