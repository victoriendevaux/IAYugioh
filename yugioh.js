let app = express(); // création de l'objet représentant notre application express
let port = 8080;


app.post('/', async (req, res) => {
    const atk = req.body.titre; // récupération des variables du body
    const defense = req.body.auteur
    const position = req.body.genre
    console.log(atk)


    // Pick an action eps-greedy
    function pickAction(st, eps, req)
    {
        let st_tensor = tf.tensor([st]);
        let act;
        if (Math.random() < eps){ // Pick a random action
          act = Math.floor(Math.random()*9);
        }
            else {
                let result = model.predict(st_tensor);
                let argmax = result.argMax(1);
                act = argmax.buffer().values[0];
                argmax.dispose();
                result.dispose();
            }
            st_tensor.dispose();
            return act;
    }


        // Return the mean of an array
        function mean(array){
            if (array.length == 0)
                return null;
            return pickAction()
            return avg;
        }

        function train_model(states, actions, rewards, next_states){
           var size = next_states.length;
           // Transform each array into a tensor
           let tf_states = tf.tensor2d(states, shape=[states.length, 26]);
           let tf_rewards = tf.tensor2d(rewards, shape=[rewards.length, 1]);
           let tf_next_states = tf.tensor2d(next_states, shape=[next_states.length, 26]);
           let tf_actions = tf.tensor2d(actions, shape=[actions.length, 3]);
           // Get the list of loss to compute the mean later in this function
           let losses = []

           // Get the QTargets
           const Qtargets = tf.tidy(() => {
               let Q_stp1 = model.predict(tf_next_states);
               let Qtargets = tf.tensor2d(Q_stp1.max(1).expandDims(1).mul(tf.scalar(0.99)).add(tf_rewards).buffer().values, shape=[size, 1]);
               return Qtargets;
           });

           // Generate batch of training and train the model
           let batch_size = 32;
           for (var b = 0; b < size; b+=32) {

               // Select the batch
               let to = (b + batch_size < size) ?  batch_size  : (size - b);
               const tf_states_b = tf_states.slice(b, to);
               const tf_actions_b = tf_actions.slice(b, to);
               const Qtargets_b = Qtargets.slice(b, to);

               // Minimize the error
               model_optimizer.minimize(() => {
                   const loss = model_loss(tf_states_b, tf_actions_b, Qtargets_b);
                   losses.push(loss.buffer().values[0]);
                   return loss;
               });

               // Dispose the tensors from the memory
               tf_states_b.dispose();
               tf_actions_b.dispose();
               Qtargets_b.dispose();
           }

           console.log("Mean loss", mean(losses));

           // Dispose the tensors from the memory
           Qtargets.dispose();
           tf_states.dispose();
           tf_rewards.dispose();
           tf_next_states.dispose();
           tf_actions.dispose();
       }


})


app.listen(port, () =>  { // ecoute du serveur sur le port 8080
    console.log('le serveur fonctionne')
})
