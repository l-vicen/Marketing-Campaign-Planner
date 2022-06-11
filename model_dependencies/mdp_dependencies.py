# Dependencies
import streamlit as st
import mdptoolbox

def solve_markov_decision_process(transition_probability, rewards, discount_factor, method, number_iterations):

    """
    solve_markov_decision_process(...) is responsable for trigering the selected MDP solver in the MDP Page
    """

    if (method == "Value Iteration"):
        model = mdptoolbox.mdp.ValueIteration(transition_probability, rewards, discount_factor)
        model.run()
        result_dict = display_simulation_results(model)
        return result_dict

    elif (method == "Policy Iteration"):
        model = mdptoolbox.mdp.PolicyIteration(transition_probability, rewards, discount_factor)
        model.run()
        result_dict = display_simulation_results(model)
        return result_dict

    elif (method == "Q-Learnings"):
        model = mdptoolbox.mdp.QLearning(transition_probability, rewards, discount_factor, number_iterations)
        model.run()
        result_dict = display_simulation_results(model)
        return result_dict

    else:
        st.warning("Please select a solver!")

def display_simulation_results(model):

    """
    display_simulation_results(...) displays the MDP results
    """

    st.markdown('---')
    st.markdown('## MDP Solution')

    c1, c2 = st.columns(2)

    result_dict = dict()
    result_dict["Value Function"] = model.V
    result_dict["Optimal Policy"] = model.policy
    result_dict["Time"] = model.time

    time = result_dict.get("Time")

    if (time > 1 and time < 5):
        st.warning('__Used CPU Time:__ {}'.format(time))
    elif (time < 1):
        st.success('__Used CPU Time:__ {}'.format(time))
    else: 
        st.error('__Used CPU Time:__ {}'.format(time))

    c1.markdown('#### Optimal Value Function')
    c1.table(result_dict.get("Value Function"))

    c2.markdown("#### Optimal Policy")
    c2.table(result_dict.get("Optimal Policy"))

    return result_dict